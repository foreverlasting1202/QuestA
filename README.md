# QuestA

<p align="center">
  | <a href="https://www.arxiv.org/abs/2507.13266"><b>Paper</b></a> | <a href="https://github.com/foreverlasting1202/QuestA/"><b>Documentation</b></a> | <a href="https://huggingface.co/foreverlasting1202/QuestA-Nemotron-1.5B"><b>🤗Models</b></a> | <a
href="https://huggingface.co/datasets/foreverlasting1202/QuestA"><b>🤗Datas</b></a> | <a
</p>

## Overview

QuestA is a reinforcement learning (RL) approach aimed at improving the reasoning capabilities of large language models (LLMs) by augmenting questions with partial solutions. This method dynamically adjusts the difficulty of problems during training by providing intermediate solutions, thereby guiding the model through difficult problems and enhancing both its efficiency and reasoning power.

This repository includes code, models, and datasets to reproduce the results presented in the associated paper.

## Repository Links

- **Code**: `./AReaL`
- **Base Model**: [Nemotron-1.5B on Hugging Face](https://huggingface.co/nvidia/OpenMath-Nemotron-1.5B)
- **Data Resource**: [OpenR1-Math-220k Dataset](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)

## Setup and Configuration

### 1. Environment Setup

1. Follow the environment configuration steps provided in the **AReaL repository** under the **AReaLite** environment setup process. Ensure that the environment is properly configured to run the QuestA experiments. You can find the official instructions in `./AReaL`.

   **Note**: Depending on your environment, you may need to modify some training script parameters (like system paths or hardware configurations) in the scripts to match your setup.

### 2. Data Preparation

1. **Download the required data**: Download the training data from [foreverlasting1202/QuestA on Hugging Face](https://huggingface.co/foreverlasting1202/QuestA).

2. **Test Data**: Choose a test dataset and process it using the same methods as the training data.

3. **Data Preprocessing**: Use the following Python scripts from the `./AReaL/datasets` directory to preprocess the data:

   - **First**, run `add_prefix.py` to add prefixes to the data.
   - **Next**, process the data using `process.py`.
   - **Finally**, convert the processed data to Hugging Face format using `convert2hf.py`.

   Apply the same processing to your chosen test dataset.

### 3. Model Training

1. Modify the environment parameters in the two YAML files found in `./AReaL/scripts` to match your system configuration.

2. To train the **Partial_50 Model**, run the following command:

   ```bash
   python3 -m arealite.launcher.ray ./AReaL/scripts/partial_50_grpo.py --config ./AReaL/scripts/partial_50_grpo.yaml
   ```

3. To train the **Partial_50_25 Model**, run:

   ```bash
   python3 -m arealite.launcher.ray ./AReaL/scripts/partial_50_25_grpo.py --config ./AReaL/scripts/partial_50_25_grpo.yaml
   ```

   **Important**: Before running this, ensure that you modify the model path in the `partial_50_25_grpo.yaml` file to point to the model obtained after training the **Partial_50 Model**.

### 4. Model Evaluation

Once you have trained the **Partial_50_25 Model**, you can evaluate it by yourself.

### 5. Pre-trained Model

We also provide a pre-trained **Partial_50_25 Model**, which you can download from:

- [Pre-trained Model](https://huggingface.co/foreverlasting1202/QuestA-Nemotron-1.5B)

## Conclusion

QuestA leverages question augmentation to efficiently train models on difficult reasoning tasks. By incorporating partial solutions during training, the model's reasoning capacity is improved, leading to substantial performance gains across math and other reasoning benchmarks. The code and models are provided to help reproduce and extend this work.

For further details on the method and experimental results, please refer to the [paper](https://www.arxiv.org/pdf/2507.13266).
