
import argparse
import json
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential
from azure.identity import ManagedIdentityCredential




def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--evaluation_output", required=True)

    parser.add_argument("--subscription_id", required=True)
    parser.add_argument("--resource_group", required=True)
    parser.add_argument("--workspace_name", required=True)

    parser.add_argument("--model_info_output_path", required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Read deploy_flag from the evaluation output folder.
    evaluation_dir = Path(args.evaluation_output)
    deploy_flag_path = evaluation_dir / "deploy_flag.txt"

    if not deploy_flag_path.exists():
        raise FileNotFoundError(
            f"deploy_flag.txt not found in {evaluation_dir}"
        )

    deploy_flag = deploy_flag_path.read_text().strip()

    # 2. Prepare the output file.
    output_path = Path(args.model_info_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if deploy_flag != "1":
        print("Model registration skipped: deploy_flag is not 1.")

        output_path.write_text(
            json.dumps(
                {
                    "registration_status": "skipped",
                    "reason": "deploy_flag is not 1",
                    "model_name": args.model_name,
                },
                indent=2,
            )
        )
        return

    # 3. Connect to the Azure ML workspace.
    #credential = DefaultAzureCredential()
    credential = ManagedIdentityCredential()

    ml_client = MLClient(
        credential=credential,
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace_name,
    )

    # 4. Register the Hugging Face model as an Azure ML custom model.
    model = Model(
        name=args.model_name,
        path=args.model_path,
        type=AssetTypes.CUSTOM_MODEL,
        description="Fine-tuned BERT model for EMR named entity recognition",
        tags={
            "framework": "transformers",
            "task": "token-classification",
        },
    )

    registered_model = ml_client.models.create_or_update(model)

    # 5. Save registration information.
    model_info = {
        "registration_status": "success",
        "model_name": registered_model.name,
        "model_version": registered_model.version,
        "model_id": registered_model.id,
        "model_type": "custom_model",
    }

    output_path.write_text(json.dumps(model_info, indent=2))

    print("Model registered successfully.")
    print(json.dumps(model_info, indent=2))


if __name__ == "__main__":
    main()
