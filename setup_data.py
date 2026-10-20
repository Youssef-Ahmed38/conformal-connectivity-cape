"""Download everything that is not in the repository (datasets, official CAPE code, SAM weights).

    python setup_data.py            # all of it (~0.5 GB)
    python setup_data.py --skip-sam # DRIVE + roads + CAPE only (phases 1-2 do not need SAM)

Layout created (identical to the one used for the paper):
    data/raw/{train,val}/{input,label}/    DRIVE (Hugging Face mirror of Staal et al. 2004; 'val' = official test set)
    data/roads/eval/{images,masks}/        Massachusetts Roads evaluation tiles (Mnih 2013)
    external/CAPE/                         official CAPE code, pinned commit (MIT)
    external/sam/sam-vit-base/             Segment Anything ViT-B weights (Apache-2.0)
Check the datasets' terms of use before redistributing them.
"""
import argparse
import subprocess
from pathlib import Path

CAPE_REPO = "https://github.com/NeuraVisionLab/CAPE.git"
CAPE_COMMIT = "e0db656998f361a97f73f2ca3df5f754b168053d"
DRIVE = "Zomba/DRIVE-digital-retinal-images-for-vessel-extraction"
ROADS = "k4nnguyen/massachusetts-roads"
SAM = "facebook/sam-vit-base"


def cape(root):
    d = root / "external" / "CAPE"
    if not (d / "loss.py").exists():
        subprocess.run(["git", "clone", CAPE_REPO, str(d)], check=True)
    subprocess.run(["git", "-C", str(d), "checkout", "-q", CAPE_COMMIT], check=True)
    print("CAPE ->", d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sam", action="store_true")
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download, snapshot_download
    root = Path(__file__).resolve().parent
    cape(root)
    snapshot_download(DRIVE, repo_type="dataset", local_dir=root / "data" / "raw", allow_patterns=["train/*", "val/*"])
    print("DRIVE ->", root / "data" / "raw")
    snapshot_download(ROADS, repo_type="dataset", local_dir=root / "data" / "roads", allow_patterns=["eval/*"])
    print("Roads ->", root / "data" / "roads")
    if not a.skip_sam:
        for f in ("config.json", "preprocessor_config.json", "model.safetensors"):
            hf_hub_download(SAM, f, local_dir=root / "external" / "sam" / "sam-vit-base")
        print("SAM   ->", root / "external" / "sam" / "sam-vit-base")


if __name__ == "__main__":
    main()
