#!/usr/bin/env python3
"""Download checkpoints from Hugging Face repos into ./checkpoints using huggingface_hub.

Usage:
  python download_checkpoints.py [--outdir checkpoints] [--token YOUR_HF_TOKEN]

It will download the files listed in the built-in manifest.
"""
import argparse
import os
import shutil
from huggingface_hub import hf_hub_download

MANIFEST = [
    # (repo_id, filename)
    ("Wan-AI/Wan2.1-T2V-1.3B", "Wan2.1_VAE.pth"),
    ("Wan-AI/Wan2.1-T2V-1.3B", "models_t5_umt5-xxl-enc-bf16.pth"),
    ("TurboDiffusion/TurboWan2.1-T2V-1.3B-480P", "TurboWan2.1-T2V-1.3B-480P.pth"),
    ("TurboDiffusion/TurboWan2.2-I2V-A14B-720P", "TurboWan2.2-I2V-A14B-high-720P.pth"),
    ("TurboDiffusion/TurboWan2.2-I2V-A14B-720P", "TurboWan2.2-I2V-A14B-low-720P.pth"),
]


def download_file(repo_id: str, filename: str, outdir: str, token: str | None):
    print(f"Downloading {filename} from {repo_id} ...")
    try:
        local_path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
    except Exception as e:
        print(f"ERROR: failed to download {filename} from {repo_id}: {e}")
        return False

    target = os.path.join(outdir, os.path.basename(filename))
    try:
        shutil.copyfile(local_path, target)
    except Exception as e:
        print(f"ERROR: failed to copy {local_path} to {target}: {e}")
        return False

    print(f"Saved -> {target}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="checkpoints", help="Output directory")
    parser.add_argument("--token", default=os.environ.get("HUGGINGFACE_TOKEN"), help="Hugging Face token (or set HUGGINGFACE_TOKEN)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    success = True
    for repo_id, filename in MANIFEST:
        ok = download_file(repo_id, filename, args.outdir, args.token)
        success = success and ok

    if not success:
        print("One or more downloads failed. Re-run with --token if any repo is private.")
        raise SystemExit(1)

    print("All downloads finished.")


if __name__ == "__main__":
    main()
