"""Upload a trained model folder (from train.py) to the Hugging Face Hub.

    export HF_TOKEN=...        # a token with write access
    python hf/push_to_hub.py --folder outputs/rs1_beattheta --repo-id <user>/RS1_BeatTheta
"""
import argparse
import os
from huggingface_hub import HfApi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folder", default="outputs/rs1_beattheta")
    p.add_argument("--repo-id", required=True)
    p.add_argument("--private", action="store_true")
    a = p.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set the HF_TOKEN environment variable to a Hugging Face token with write access.")
    api = HfApi(token=token)
    api.create_repo(a.repo_id, repo_type="model", private=a.private, exist_ok=True)
    api.upload_folder(folder_path=a.folder, repo_id=a.repo_id, repo_type="model",
                      ignore_patterns=["train_state.pt", "*.tmp", "eval/*"], commit_message="Add RS1_BeatTheta")
    print(f"Uploaded to https://huggingface.co/{a.repo_id}")


if __name__ == "__main__":
    main()
