
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw_data",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--prepared_dataset",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="google-bert/bert-base-uncased",
    )

    return parser.parse_args()


def main(args):

    from datasets import load_dataset
    from transformers import AutoTokenizer

    # -------------------------------------------------
    # 1. Entity labels
    # -------------------------------------------------

    label_list = [
        "O",
        "B-ADMISSION_DATE_KEY",
        "I-ADMISSION_DATE_KEY",
        "B-ADMISSION_DATE_VALUE",
        "I-ADMISSION_DATE_VALUE",
        "B-BP_KEY",
        "I-BP_KEY",
        "B-BP_VALUE",
        "I-BP_VALUE",
        "B-TEMPERATURE_KEY",
        "I-TEMPERATURE_KEY",
        "B-TEMPERATURE_VALUE",
        "I-TEMPERATURE_VALUE",
        "B-DISCHARGE_DATE_KEY",
        "I-DISCHARGE_DATE_KEY",
        "B-DISCHARGE_DATE_VALUE",
        "I-DISCHARGE_DATE_VALUE",
        "B-DISCHARGE_DISPOSITION_KEY",
        "I-DISCHARGE_DISPOSITION_KEY",
        "B-DISCHARGE_DISPOSITION_VALUE",
        "I-DISCHARGE_DISPOSITION_VALUE",
    ]

    label2id = {
        label: index
        for index, label in enumerate(label_list)
    }

    # -------------------------------------------------
    # 2. Input and output directories
    # -------------------------------------------------

    data_dir = Path(args.raw_data)
    output_dir = Path(args.prepared_dataset)

    output_dir.mkdir(parents=True, exist_ok=True)

    data_files = {
        "train": str(data_dir / "train_chunks.jsonl"),
        "validation": str(data_dir / "validation_chunks.jsonl"),
        "test": str(data_dir / "test_chunks.jsonl"),
    }

    for split, file_path in data_files.items():
        if not Path(file_path).exists():
            raise FileNotFoundError(
                f"Missing {split} dataset: {file_path}"
            )

    # -------------------------------------------------
    # 3. Load datasets
    # -------------------------------------------------

    dataset = load_dataset(
        "json",
        data_files=data_files,
    )

    # -------------------------------------------------
    # 4. Validate and prepare examples
    # -------------------------------------------------

    def prepare_example(example):

        tokens = example["word_tokens"]
        bio_tags = example["word_bio_tags"]

        if len(tokens) != len(bio_tags):
            raise ValueError(
                f"Token/label mismatch in {example['sample_id']}"
            )

        unknown_labels = set(bio_tags) - set(label2id)

        if unknown_labels:
            raise ValueError(
                f"Unknown labels: {unknown_labels}"
            )

        return {
            "tokens": tokens,
            "ner_tags": [
                label2id[tag]
                for tag in bio_tags
            ],
        }

    for split in dataset.keys():

        columns_to_remove = [
            column
            for column in dataset[split].column_names
            if column not in {"sample_id", "document_id"}
        ]

        dataset[split] = dataset[split].map(
            prepare_example,
            remove_columns=columns_to_remove,
        )

        print(f"{split}: {len(dataset[split])} examples")

    # -------------------------------------------------
    # 5. Load tokenizer
    # -------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_checkpoint
    )

    # -------------------------------------------------
    # 6. Tokenization and label alignment
    # -------------------------------------------------

    def tokenize_and_align_labels(examples):

        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
        )

        aligned_labels = []

        for batch_index, word_labels in enumerate(
            examples["ner_tags"]
        ):

            word_ids = tokenized_inputs.word_ids(
                batch_index=batch_index
            )

            previous_word_id = None
            label_ids = []

            for word_id in word_ids:

                if word_id is None:
                    label_ids.append(-100)

                elif word_id != previous_word_id:
                    label_ids.append(word_labels[word_id])

                else:
                    label_ids.append(-100)

                previous_word_id = word_id

            aligned_labels.append(label_ids)

        tokenized_inputs["labels"] = aligned_labels

        return tokenized_inputs

    tokenized_dataset = dataset.map(
        tokenize_and_align_labels,
        batched=True,
    )

    # -------------------------------------------------
    # 7. Save prepared dataset
    # -------------------------------------------------

    tokenized_dataset.save_to_disk(
        str(output_dir)
    )

    print("Dataset preparation completed.")

    for split in tokenized_dataset:
        print(
            f"{split}: {len(tokenized_dataset[split])} examples"
        )

    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
