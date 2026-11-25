
import argparse
import json
import time
import subprocess
from pathlib import Path

import psutil


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune BERT for EMR named entity recognition"
    )

    parser.add_argument(
        "--prepared_dataset",
        type=str,
        required=True,
        help="Path to the prepared tokenized dataset",
    )

    parser.add_argument(
        "--model_output",
        type=str,
        required=True,
        help="Output directory for the trained model",
    )

    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="google-bert/bert-base-uncased",
    )

    return parser.parse_args()


def main(args):

    import torch
    import numpy as np
    import evaluate

    from datasets import load_from_disk

    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
        TrainerCallback,
    )

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

    id2label = {
        index: label
        for label, index in label2id.items()
    }

    # -------------------------------------------------
    # 2. Detect CPU or GPU
    # -------------------------------------------------

    print("\n========== HARDWARE DIAGNOSTICS ==========")

    print("PyTorch version:", torch.__version__)
    print("PyTorch CUDA version:", torch.version.cuda)

    use_gpu = torch.cuda.is_available()

    print("CUDA available:", use_gpu)

    if use_gpu:

        print("Selected device: GPU")
        print("GPU count:", torch.cuda.device_count())

        for index in range(torch.cuda.device_count()):

            properties = torch.cuda.get_device_properties(index)

            print(f"\nGPU {index}:")
            print("Name:", properties.name)

            print(
                "Compute capability:",
                properties.major,
                properties.minor,
            )

            print(
                "Total GPU memory:",
                round(properties.total_memory / 1024**3, 2),
                "GB",
            )

        print("\nNVIDIA-SMI output:")

        subprocess.run(
            ["nvidia-smi"],
            check=False,
        )

    else:

        print("Selected device: CPU")
        print("No CUDA-compatible GPU detected.")
        print("Training will run on CPU.")

    print("CPU count:", psutil.cpu_count(logical=True))
    print(
        "System RAM:",
        round(psutil.virtual_memory().total / 1024**3, 2),
        "GB",
    )

    print("==========================================\n")

    # -------------------------------------------------
    # 3. Load prepared dataset
    # -------------------------------------------------

    dataset = load_from_disk(str(args.prepared_dataset))

    for split in dataset:
        print(f"{split}: {len(dataset[split])} examples")

    train_dataset = dataset["train"]
    validation_dataset = dataset["validation"]

    # -------------------------------------------------
    # 4. Load tokenizer and model
    # -------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_checkpoint
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_checkpoint,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    print(
        "\nInitial model device:",
        next(model.parameters()).device,
    )

    # -------------------------------------------------
    # 5. Data collator
    # -------------------------------------------------

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer
    )

    # -------------------------------------------------
    # 6. Evaluation metric
    # -------------------------------------------------

    seqeval = evaluate.load("seqeval")

    def compute_metrics(eval_prediction):

        logits, labels = eval_prediction

        if isinstance(logits, tuple):
            logits = logits[0]

        predictions = np.argmax(logits, axis=2)

        true_predictions = []
        true_labels = []

        for prediction, label in zip(predictions, labels):

            sample_predictions = []
            sample_labels = []

            for predicted_id, label_id in zip(
                prediction,
                label,
            ):

                if label_id == -100:
                    continue

                sample_predictions.append(
                    label_list[int(predicted_id)]
                )

                sample_labels.append(
                    label_list[int(label_id)]
                )

            true_predictions.append(sample_predictions)
            true_labels.append(sample_labels)

        results = seqeval.compute(
            predictions=true_predictions,
            references=true_labels,
        )

        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    # -------------------------------------------------
    # 7. Resource monitoring callback
    # -------------------------------------------------

    class ResourceMonitorCallback(TrainerCallback):

        def __init__(self):

            self.process = psutil.Process()

            # Initialize CPU usage measurement.
            self.process.cpu_percent(interval=None)
            psutil.cpu_percent(interval=None)

        def on_log(self, args, state, control, logs=None, **kwargs):

            print("\n========== RESOURCE USAGE ==========")
            print("Step:", state.global_step)

            # CPU utilization
            process_cpu = self.process.cpu_percent(interval=None)
            system_cpu = psutil.cpu_percent(interval=None)

            print(f"Process CPU usage: {process_cpu:.2f}%")
            print(f"System CPU usage: {system_cpu:.2f}%")

            # RAM usage
            process_memory = self.process.memory_info().rss
            system_memory = psutil.virtual_memory()

            print(
                "Process RAM usage:",
                round(process_memory / 1024**3, 2),
                "GB",
            )

            print(
                "System RAM usage:",
                round(system_memory.used / 1024**3, 2),
                "GB",
            )

            print(
                "System RAM utilization:",
                f"{system_memory.percent:.2f}%",
            )

            # GPU monitoring
            if torch.cuda.is_available():

                allocated = (
                    torch.cuda.memory_allocated(0) / 1024**3
                )

                reserved = (
                    torch.cuda.memory_reserved(0) / 1024**3
                )

                peak_allocated = (
                    torch.cuda.max_memory_allocated(0) / 1024**3
                )

                print("\nGPU:", torch.cuda.get_device_name(0))

                print(
                    f"GPU memory allocated: {allocated:.2f} GB"
                )

                print(
                    f"GPU memory reserved: {reserved:.2f} GB"
                )

                print(
                    f"Peak GPU memory allocated: {peak_allocated:.2f} GB"
                )

                try:

                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                    )

                    print(
                        "GPU utilization and memory used:",
                        result.stdout.strip(),
                    )

                except (
                    FileNotFoundError,
                    subprocess.CalledProcessError,
                ):

                    print("GPU utilization metrics unavailable.")

            print("====================================\n")

    # -------------------------------------------------
    # 8. Training arguments
    # -------------------------------------------------

    training_args = TrainingArguments(

        output_dir="/tmp/emr-ner-training",

        learning_rate=args.learning_rate,

        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,

        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,

        eval_strategy="epoch",
        save_strategy="epoch",

        logging_strategy="steps",
        logging_steps=10,

        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        save_total_limit=2,

        # Automatically select CPU or GPU.
        use_cpu=not use_gpu,

        report_to="none",
    )

    # -------------------------------------------------
    # 9. Initialize Trainer
    # -------------------------------------------------

    trainer = Trainer(

        model=model,
        args=training_args,

        train_dataset=train_dataset,
        eval_dataset=validation_dataset,

        processing_class=tokenizer,
        data_collator=data_collator,

        compute_metrics=compute_metrics,

        callbacks=[
            ResourceMonitorCallback()
        ],
    )

    # -------------------------------------------------
    # 10. Verify actual Trainer device
    # -------------------------------------------------

    print("\n========== TRAINER DEVICE DIAGNOSTICS ==========")

    print("Trainer device:", training_args.device)

    print(
        "Trainer model device:",
        next(trainer.model.parameters()).device,
    )

    print("FP16:", training_args.fp16)
    print("BF16:", training_args.bf16)

    if use_gpu:

        print(
            "GPU allocated memory:",
            round(torch.cuda.memory_allocated(0) / 1024**3, 3),
            "GB",
        )

        print(
            "GPU reserved memory:",
            round(torch.cuda.memory_reserved(0) / 1024**3, 3),
            "GB",
        )

    print("================================================\n")

    # -------------------------------------------------
    # 11. Train with timing and resource monitoring
    # -------------------------------------------------

    if use_gpu:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    training_start = time.perf_counter()

    print("\n========== TRAINING STARTED ==========\n")

    train_result = trainer.train()

    if use_gpu:
        torch.cuda.synchronize()

    training_duration = time.perf_counter() - training_start

    print("\n========== TRAINING COMPLETED ==========\n")

    print(
        "Training duration:",
        round(training_duration, 2),
        "seconds",
    )

    print("Training metrics:")
    print(train_result.metrics)

    # -------------------------------------------------
    # 12. Final resource usage summary
    # -------------------------------------------------

    print("\n========== FINAL RESOURCE SUMMARY ==========")

    process = psutil.Process()
    system_memory = psutil.virtual_memory()

    print("Selected device:", training_args.device)

    print(
        "Process RAM usage:",
        round(process.memory_info().rss / 1024**3, 3),
        "GB",
    )

    print(
        "System RAM utilization:",
        f"{system_memory.percent:.2f}%",
    )

    print(
        "System CPU utilization:",
        f"{psutil.cpu_percent(interval=1):.2f}%",
    )

    if use_gpu:

        print(
            "Current GPU allocated memory:",
            round(torch.cuda.memory_allocated(0) / 1024**3, 3),
            "GB",
        )

        print(
            "Peak GPU allocated memory:",
            round(torch.cuda.max_memory_allocated(0) / 1024**3, 3),
            "GB",
        )

        print(
            "Current GPU reserved memory:",
            round(torch.cuda.memory_reserved(0) / 1024**3, 3),
            "GB",
        )

        print(
            "Peak GPU reserved memory:",
            round(torch.cuda.max_memory_reserved(0) / 1024**3, 3),
            "GB",
        )

    print("============================================\n")

    # -------------------------------------------------
    # 13. Save best model and tokenizer
    # -------------------------------------------------

    output_path = Path(args.model_output)
    output_path.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    print(
        "Best checkpoint:",
        trainer.state.best_model_checkpoint,
    )

    print(
        "Best validation metric:",
        trainer.state.best_metric,
    )

    print("Model saved to:", output_path)


if __name__ == "__main__":

    args = parse_args()

    print("Prepared dataset path:", args.prepared_dataset)
    print("Model output path:", args.model_output)
    print("Learning rate:", args.learning_rate)
    print("Batch size:", args.batch_size)
    print("Epochs:", args.epochs)
    print("Weight decay:", args.weight_decay)
    print("Model checkpoint:", args.model_checkpoint)

    main(args)
