
import argparse
import json
from pathlib import Path

import evaluate
import mlflow
import numpy as np
import torch

from datasets import load_from_disk
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)


# This mapping must exactly match prepare_dataset.py and train_model.py.
LABELS = [
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

FALLBACK_ID2LABEL = {
    i: label for i, label in enumerate(LABELS)
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_input", required=True)
    parser.add_argument("--prepared_dataset", required=True)
    parser.add_argument("--evaluation_output", required=True)
    parser.add_argument("--batch_size", type=int, default=8)

    return parser.parse_args()


# Recursively convert NumPy values into JSON-serializable Python values.
def json_safe(value):

    if isinstance(value, dict):
        return {
            key: json_safe(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    return value


def main():

    args = parse_args()

    output_dir = Path(args.evaluation_output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 1. Load the prepared test dataset
    # --------------------------------------------------

    dataset = load_from_disk(args.prepared_dataset)

    if "test" not in dataset:
        raise ValueError(
            f"No 'test' split found. "
            f"Available splits: {list(dataset.keys())}"
        )

    test_dataset = dataset["test"]

    required_columns = {
        "input_ids",
        "attention_mask",
        "labels",
    }

    missing = required_columns.difference(
        test_dataset.column_names
    )

    if missing:
        raise ValueError(
            f"Test dataset is missing columns: {sorted(missing)}"
        )

    print(f"Test examples: {len(test_dataset)}")

    # --------------------------------------------------
    # 2. Load the trained Hugging Face model
    # --------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_input
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_input
    )

    if model.config.num_labels != len(LABELS):
        raise ValueError(
            f"Model has {model.config.num_labels} labels, "
            f"but evaluation mapping contains {len(LABELS)}."
        )

    # Prefer the label mapping saved in the model configuration.
    configured_mapping = model.config.id2label

    id2label = {
        int(key): value
        for key, value in configured_mapping.items()
    }

    if (
        len(id2label) != len(LABELS)
        or any(i not in id2label for i in range(len(LABELS)))
    ):
        id2label = FALLBACK_ID2LABEL

    # --------------------------------------------------
    # 3. Configure evaluation
    # --------------------------------------------------

    use_gpu = torch.cuda.is_available()

    print(f"GPU available: {use_gpu}")

    training_args = TrainingArguments(
        output_dir=str(output_dir / "trainer_tmp"),
        per_device_eval_batch_size=args.batch_size,
        report_to="none",
        use_cpu=not use_gpu,
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    # --------------------------------------------------
    # 4. Run predictions on the test dataset
    # --------------------------------------------------

    print("Running predictions on test dataset...")

    prediction_result = trainer.predict(test_dataset)

    predicted_ids = np.argmax(
        prediction_result.predictions,
        axis=-1,
    )

    true_ids = prediction_result.label_ids

    # --------------------------------------------------
    # 5. Convert token IDs into BIO labels
    # --------------------------------------------------

    seqeval = evaluate.load("seqeval")

    true_sequences = []
    predicted_sequences = []

    correct_tokens = 0
    total_tokens = 0

    for true_row, predicted_row in zip(
        true_ids,
        predicted_ids,
    ):

        true_tags = []
        predicted_tags = []

        for true_id, predicted_id in zip(
            true_row,
            predicted_row,
        ):

            # Ignore special tokens, padding and continuation subtokens.
            if int(true_id) == -100:
                continue

            true_id = int(true_id)
            predicted_id = int(predicted_id)

            if (
                true_id not in id2label
                or predicted_id not in id2label
            ):
                raise ValueError(
                    f"Unknown label ID: "
                    f"true={true_id}, predicted={predicted_id}"
                )

            true_tags.append(id2label[true_id])
            predicted_tags.append(id2label[predicted_id])

            correct_tokens += int(
                true_id == predicted_id
            )

            total_tokens += 1

        true_sequences.append(true_tags)
        predicted_sequences.append(predicted_tags)

    if total_tokens == 0:
        raise ValueError(
            "No evaluable tokens found in the test dataset."
        )

    # --------------------------------------------------
    # 6. Calculate NER evaluation metrics
    # --------------------------------------------------

    report = seqeval.compute(
        predictions=predicted_sequences,
        references=true_sequences,
    )

    overall = {
        "precision": float(report["overall_precision"]),
        "recall": float(report["overall_recall"]),
        "f1": float(report["overall_f1"]),
        "token_accuracy": float(
            correct_tokens / total_tokens
        ),
        "evaluated_tokens": int(total_tokens),
        "evaluated_examples": int(len(test_dataset)),
    }

    per_entity = {
        key: value
        for key, value in report.items()
        if not key.startswith("overall_")
    }

    print("Evaluation metrics calculated.")

    # --------------------------------------------------
    # 7. Save metrics.json
    # --------------------------------------------------

    metrics_data = json_safe({
        "overall": overall,
        "per_entity": per_entity,
    })

    with open(
        output_dir / "metrics.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics_data,
            f,
            indent=2,
        )

    # --------------------------------------------------
    # 8. Save classification report
    # --------------------------------------------------

    lines = [
        "EMR NER Evaluation",
        "==================",
        "",
        "Overall Metrics",
        "---------------",
    ]

    for key, value in overall.items():
        lines.append(f"{key}: {value}")

    lines.extend([
        "",
        "Per-Entity Metrics",
        "------------------",
    ])

    for entity, values in per_entity.items():

        lines.append(
            f"{entity}: "
            f"precision={values['precision']:.4f}, "
            f"recall={values['recall']:.4f}, "
            f"f1={values['f1']:.4f}"
        )

    with open(
        output_dir / "classification_report.txt",
        "w",
        encoding="utf-8",
    ) as f:

        f.write("\n".join(lines) + "\n")

    # --------------------------------------------------
    # 9. Save predictions.jsonl
    # --------------------------------------------------

    id_columns = [
        column
        for column in ("sample_id", "document_id")
        if column in test_dataset.column_names
    ]

    with open(
        output_dir / "predictions.jsonl",
        "w",
        encoding="utf-8",
    ) as f:

        for index, (actual, predicted) in enumerate(
            zip(true_sequences, predicted_sequences)
        ):

            row = {
                "example_index": index,
            }

            for column in id_columns:
                row[column] = json_safe(
                    test_dataset[index][column]
                )

            row["true_tags"] = actual
            row["predicted_tags"] = predicted

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                ) + "\n"
            )

    # --------------------------------------------------
    # 10. Set deploy_flag = 1
    # --------------------------------------------------

    # Model promotion comparison is intentionally removed.
    # Successful evaluation always produces deploy_flag = 1.

    deploy_flag_path = output_dir / "deploy_flag.txt"

    deploy_flag_path.write_text(
        "1\n",
        encoding="utf-8",
    )

    print("deploy_flag = 1")

    # --------------------------------------------------
    # 11. Log metrics and artifacts to MLflow
    # --------------------------------------------------

    with mlflow.start_run():

        mlflow.log_metrics(overall)

        mlflow.log_artifacts(
            str(output_dir),
            artifact_path="evaluation",
        )

    # --------------------------------------------------
    # 12. Final summary
    # --------------------------------------------------

    print("\nEvaluation completed successfully.")

    print(
        json.dumps(
            metrics_data,
            indent=2,
        )
    )

    print(f"\nEvaluation artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
