from datasets import load_dataset

if __name__ == "__main__":
    ds = load_dataset(
        "json",
        data_files={
            "train": "data/train.jsonl",
            "test": "data/test.jsonl"
        }
    )

    print(ds)

    ds.save_to_disk("openr1_50")