# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import copy
import dataclasses
import os
import pickle
import shutil
from typing import Dict

import torch
import torch.distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizerFast

from arealite.api.cli_args import RecoverConfig
from arealite.api.engine_api import InferenceEngine, TrainEngine
from arealite.api.io_struct import FinetuneSpec, SaveLoadMeta, WeightUpdateMeta
from arealite.api.workflow_api import RolloutWorkflow
from arealite.utils.evaluator import Evaluator
from arealite.utils.saver import Saver
from arealite.utils.stats_logger import StatsLogger
from realhf.base import logging, timeutil

logger = logging.getLogger("recover")


@dataclasses.dataclass
class StepInfo:
    epoch: int
    epoch_step: int
    global_step: int
    steps_per_epoch: int

    def next(self):
        return StepInfo(
            epoch=self.epoch + (self.epoch_step == self.steps_per_epoch - 1),
            epoch_step=(
                0
                if self.epoch_step == self.steps_per_epoch - 1
                else self.epoch_step + 1
            ),
            global_step=self.global_step + 1,
            steps_per_epoch=self.steps_per_epoch,
        )


@dataclasses.dataclass
class RecoverInfo:
    # Recover start is the counter of the next RLHF interation
    # w.r.t. the counter of the saved checkpoint
    recover_start: StepInfo
    # Last step info is the counter of the saved checkpoint.
    # It exactly lags beind recover_start by 1 iteration.
    last_step_info: StepInfo

    saver_info: Dict
    evaluator_info: Dict
    stats_logger_info: Dict
    dataloader_info: Dict
    checkpoint_info: Dict
    inference_engine_info: Dict | None = dataclasses.field(default_factory=dict)


class InValidRecoverInfo(Exception):
    pass


class RecoverHandler:
    def __init__(self, config: RecoverConfig, ft_spec: FinetuneSpec):
        self.config = config
        self.ft_spec = ft_spec
        self.last_step_info = StepInfo(
            epoch=-1,
            epoch_step=-1,
            global_step=-1,
            steps_per_epoch=ft_spec.steps_per_epoch,
        )
        self.freq_ctl = timeutil.EpochStepTimeFreqCtl(
            freq_epoch=config.freq_epochs,
            freq_step=config.freq_steps,
            freq_sec=config.freq_secs,
        )

    @staticmethod
    def recover_info_path(
        experiment_name: str,
        trial_name: str,
        fileroot: str,
    ):
        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
        return os.path.join(
            Saver.get_save_root(experiment_name, trial_name, fileroot),
            f"recover_info_rank{rank}.pkl",
        )

    def dump(
        self,
        engine: TrainEngine,
        step_info: StepInfo,
        saver: Saver,
        evaluator: Evaluator,
        stats_logger: StatsLogger,
        dataloader: StatefulDataLoader,
        inference_engine: InferenceEngine | None = None,
    ):
        # currently only support recover on one engine
        if not self.freq_ctl.check(
            epochs=int(step_info.epoch_step == self.ft_spec.steps_per_epoch - 1),
            steps=1,
        ):
            return
        self._save_checkpoint(engine, step_info)

        recover_info = RecoverInfo(
            recover_start=self.last_step_info.next(),
            last_step_info=self.last_step_info,
            saver_info=saver.state_dict(),
            evaluator_info=evaluator.state_dict(),
            stats_logger_info=stats_logger.state_dict(),
            dataloader_info=dataloader.state_dict(),
            checkpoint_info=self.freq_ctl.state_dict(),
            inference_engine_info=(
                inference_engine.state_dict() if inference_engine else {}
            ),
        )

        recover_info_path = self.recover_info_path(
            self.config.experiment_name,
            self.config.trial_name,
            self.config.fileroot,
        )
        os.makedirs(os.path.dirname(recover_info_path), exist_ok=True)
        with open(recover_info_path, "wb") as f:
            print(f"dumping recover info to {recover_info_path}: {recover_info}")
            pickle.dump(recover_info, f)

    def load(
        self,
        engine: TrainEngine,
        saver: Saver,
        evaluator: Evaluator,
        stats_logger: StatsLogger,
        dataloader: StatefulDataLoader,
        inference_engine: InferenceEngine | None = None,
        weight_update_meta: WeightUpdateMeta | None = None,
        workflow: RolloutWorkflow | None = None,
    ):
        if os.environ.get("AREAL_RECOVER_RUN", "0") != "1":
            return
        if inference_engine is not None:
            assert (
                weight_update_meta is not None
            ), "Inference engine requires weight update meta for recovery."
            assert (
                workflow is not None
            ), "Inference engine requires a workflow for recovery."

        recover_info_path = self.recover_info_path(
            self.config.experiment_name,
            self.config.trial_name,
            self.config.fileroot,
        )
        logger.info(f"Loading recover info from {recover_info_path}")
        os.makedirs(os.path.dirname(recover_info_path), exist_ok=True)
        try:
            with open(recover_info_path, "rb") as f:
                recover_info: RecoverInfo = pickle.load(f)
                logger.info(f"Recovering from {recover_info.recover_start}")
            saver.load_state_dict(recover_info.saver_info)
            self.freq_ctl.load_state_dict(recover_info.checkpoint_info)
            evaluator.load_state_dict(recover_info.evaluator_info)
            stats_logger.load_state_dict(recover_info.stats_logger_info)
            dataloader.load_state_dict(recover_info.dataloader_info)

            self._load_checkpoint(engine)
            global_step = recover_info.last_step_info.global_step

            if inference_engine is not None:
                # update inference engine weights
                engine.set_version(global_step)
                inference_engine.set_version(global_step)
                inference_engine.pause()
                if dist.get_rank() == 0:
                    future = inference_engine.update_weights(weight_update_meta)
                engine.upload_weights(weight_update_meta)
                if dist.get_rank() == 0:
                    future.result()
                dist.barrier(device_ids=[engine.device.index])
                torch.cuda.synchronize()
                inference_engine.resume()
                engine.set_version(global_step + 1)
                inference_engine.set_version(global_step + 1)

                # submit data for inference engine according to state
                recover_data_loader = copy.deepcopy(dataloader)
                inference_engine.recover(
                    recover_info.inference_engine_info, recover_data_loader, workflow
                )

            return recover_info
        except FileNotFoundError:
            logger.warning(
                f"Resume info not found at {recover_info_path}. "
                f"This should not be a resumed experiment!"
            )

    def _save_checkpoint(
        self,
        engine: TrainEngine,
        step_info: StepInfo,
        name: str = "default",
        tokenizer: PreTrainedTokenizerFast | None = None,
        base_model_path: str | None = None,
    ):
        path = os.path.join(
            Saver.get_save_checkpoint_root(
                self.config.experiment_name,
                self.config.trial_name,
                self.config.fileroot,
                name,
            ),
            "recover_checkpoint",
        )
        # remove previous checkpoint if exists
        # try:
        #     if dist.get_rank() == 0:
        #         shutil.rmtree(path)
        # except FileNotFoundError:
        #     pass
        weight_format = "hf"
        with_optim = True
        meta = SaveLoadMeta(
            path=path,
            weight_format=weight_format,
            with_optim=with_optim,
            tokenizer=tokenizer,
            base_model_path=base_model_path,
        )
        engine.save(meta)
        print(f"saved recover checkpoint to {path}")
        self.last_step_info = step_info

    def _load_checkpoint(
        self,
        engine: TrainEngine,
        name: str = "default",
        tokenizer: PreTrainedTokenizerFast | None = None,
        base_model_path: str | None = None,
    ):
        path = os.path.join(
            Saver.get_save_checkpoint_root(
                self.config.experiment_name,
                self.config.trial_name,
                self.config.fileroot,
                name,
            ),
            "recover_checkpoint",
        )
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint path {path} does not exist.")
        weight_format = "hf"
        with_optim = True
        meta = SaveLoadMeta(
            path=path,
            weight_format=weight_format,
            with_optim=with_optim,
            tokenizer=tokenizer,
            base_model_path=base_model_path,
        )
        engine.load(meta)


def check_if_auto_recover(
    config: RecoverConfig,
) -> bool:
    # This method is called only by launchers to check if the experiment should be a recover run
    # when "recover_mode" is auto.
    experiment_name = config.experiment_name
    trial_name = config.trial_name
    fileroot = config.fileroot
    recover_info_file = RecoverHandler.recover_info_path(
        experiment_name, trial_name, fileroot
    )
    print(f"recover info file = {recover_info_file}", flush=True)
    if os.path.exists(str(recover_info_file)):
        print(f"recover info file exists", flush=True)
        with open(recover_info_file, "rb") as f:
            info: RecoverInfo = pickle.load(f)
            print(f"recover info: {info}", flush=True)
        if info.last_step_info.epoch < 0:
            msg = (
                f"Recover checkpoint is not valid. "
                f"Expected last_step_info.epoch >= 0, "
                f"but found {info.last_step_info.epoch}"
            )
            logger.warning(msg)
            return False

        save_root = Saver.get_save_root(experiment_name, trial_name, fileroot)
        for name in os.listdir(save_root):
            if not os.path.isdir(os.path.join(save_root, name)):
                continue
            path = os.path.join(
                Saver.get_save_checkpoint_root(
                    experiment_name, trial_name, fileroot, name
                ),
                "recover_checkpoint",
            )
            print(f"save root={save_root} path={path}", flush=True)
            if not os.path.exists(path):
                print(f"{path} not exists!!!", flush=True)
                logger.warning(f"Recover checkpoint for model {name} does not exist.")
                return False
        return True
    logger.warning(f"Recover info not found at: {recover_info_file}")
    return False


def check_if_recover(config: RecoverConfig, run_id: int) -> bool:
    # This method is called by the launcher to check if the experiment should be a recover run
    # when "recover_mode" is not disabled.
    if config.mode == "disabled":
        return False
    elif config.mode == "auto":
        return check_if_auto_recover(config)
    elif config.mode == "fault":
        return run_id > 0
    elif config.mode == "resume":
        return True
    else:
        raise ValueError(f"Unknown recover mode: {config.mode}")
