#!/usr/bin/env python3
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("DIGER_DATA_ROOT", ROOT / "dataset")).expanduser().resolve()
RQVAE_ROOT = Path(os.environ.get("DIGER_RQVAE_ROOT", ROOT / "rqvae_ckpt")).expanduser().resolve()

EXPECTED = {
    "beauty": {
        "embedding": "Beauty.emb-llama.npy",
        "shape": (12101, 4096),
        "checkpoint_sha256": "d4501128dd9b2db3072376f66bbaa42995567b94c88c9befab0c349fa9a242ff",
        "map_items": 12102,
        "splits": {"train": 131413, "valid": 22363, "test": 22363},
    },
    "instruments": {
        "embedding": "Instruments.emb-llama.npy",
        "shape": (9922, 4096),
        "checkpoint_sha256": "c0e435563e521e18b6b918cdf663485dedb994f6275210a8074de82db02a6b91",
        "map_items": 9923,
        "splits": {"train": 131837, "valid": 24772, "test": 24772},
    },
    "yelp": {
        "embedding": "Yelp.emb-llama.npy",
        "shape": (20033, 4096),
        "checkpoint_sha256": "c8654acb6297e67dec73eed58cb4e7b464dc5a2dd51006068e36d95d07f56bb6",
        "map_items": 20034,
        "splits": {"train": 225061, "valid": 30431, "test": 30431},
    },
}


def is_lfs_pointer(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(64).startswith(b"version https://git-lfs.github.com/spec/v1")


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def check_file(path: Path, errors: list[str]) -> bool:
    if not path.exists():
        errors.append(f"missing file: {display_path(path)}")
        return False
    if is_lfs_pointer(path):
        errors.append(f"LFS pointer was not pulled: {display_path(path)}")
        return False
    return True


def main() -> int:
    errors: list[str] = []

    for dataset, expected in EXPECTED.items():
        error_count = len(errors)
        dataset_dir = DATA_ROOT / dataset
        map_path = dataset_dir / f"{dataset}.emb_map.json"
        emb_path = dataset_dir / expected["embedding"]
        ckpt_path = RQVAE_ROOT / dataset / "best_collision_model.pth"

        if check_file(map_path, errors):
            with map_path.open() as handle:
                map_items = len(json.load(handle))
            if map_items != expected["map_items"]:
                errors.append(f"{dataset}: map has {map_items} items, expected {expected['map_items']}")

        if check_file(emb_path, errors):
            emb = np.load(emb_path, mmap_mode="r")
            if tuple(emb.shape) != expected["shape"]:
                errors.append(f"{dataset}: embedding shape is {tuple(emb.shape)}, expected {expected['shape']}")

        if check_file(ckpt_path, errors):
            if ckpt_path.stat().st_size < 100_000_000:
                errors.append(f"{dataset}: checkpoint is unexpectedly small: {ckpt_path.stat().st_size} bytes")
            actual_sha256 = sha256_file(ckpt_path)
            if actual_sha256 != expected["checkpoint_sha256"]:
                errors.append(
                    f"{dataset}: checkpoint SHA256 is {actual_sha256}, "
                    f"expected {expected['checkpoint_sha256']}"
                )

        for split, expected_lines in expected["splits"].items():
            split_path = dataset_dir / f"{dataset}.{split}.jsonl"
            if check_file(split_path, errors):
                actual_lines = count_lines(split_path)
                if actual_lines != expected_lines:
                    errors.append(f"{dataset}.{split}: {actual_lines} rows, expected {expected_lines}")

        if len(errors) == error_count:
            print(
                f"{dataset}: ok "
                f"items={expected['map_items']} "
                f"embedding={expected['shape']} "
                f"splits={expected['splits']}"
            )
        else:
            print(f"{dataset}: failed")

    if errors:
        print("\nArtifact check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("\nAll released datasets, embeddings, and RQ-VAE checkpoints are present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
