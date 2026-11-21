"""Automated dataset discovery, readability, mask alignment, and split-leakage validation CLI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from src.data.adapters import DATASET_ADAPTERS, get_dataset_adapter


def validate_dataset(dataset_id: str, data_dir: str | Path) -> Dict[str, Any]:
    adapter = get_dataset_adapter(dataset_id, data_dir)
    categories = adapter.list_categories()
    report: Dict[str, Any] = {
        "dataset": dataset_id,
        "name": adapter.metadata.name,
        "license": adapter.metadata.license,
        "root_dir": str(Path(data_dir).resolve()),
        "available": len(categories) > 0,
        "categories": {},
    }

    if not categories:
        report["note"] = (
            f"No category folders detected under {data_dir}. Follow docs/DATASETS.md to obtain "
            f"official archives from {adapter.metadata.official_url}."
        )
        return report

    try:
        from PIL import Image
    except ImportError:
        Image = None  # type: ignore

    for cat in categories:
        splits = adapter.load_category(cat)
        train_paths = {s.image_path for s in splits["train"]}
        val_paths = {s.image_path for s in splits["val"]}
        test_paths = {s.image_path for s in splits["test"]}
        leakage = bool((train_paths & test_paths) or (val_paths & test_paths))

        readable = 0
        corrupted = 0
        mask_aligned = True
        missing_masks = 0

        for split_records in splits.values():
            for rec in split_records:
                if Image is None:
                    readable += 1
                    continue
                try:
                    with Image.open(rec.image_path) as im:
                        w, h = im.size
                    readable += 1
                    if rec.label == 1:
                        if rec.mask_path and Path(rec.mask_path).exists():
                            with Image.open(rec.mask_path) as mk:
                                if mk.size != (w, h):
                                    mask_aligned = False
                        else:
                            missing_masks += 1
                except Exception:
                    corrupted += 1

        report["categories"][cat] = {
            "train_normal": len(splits["train"]),
            "val_normal": len(splits["val"]),
            "test_normal": sum(1 for s in splits["test"] if s.label == 0),
            "test_anomalous": sum(1 for s in splits["test"] if s.label == 1),
            "readable_images": readable,
            "corrupted_images": corrupted,
            "missing_masks": missing_masks,
            "masks_aligned": mask_aligned,
            "split_leakage_check": "failed" if leakage else "passed",
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="VisionGuard Dataset Validation CLI")
    parser.add_argument("--dataset", choices=list(DATASET_ADAPTERS.keys()), default="mvtec_ad")
    parser.add_argument("--data-dir", default="data/raw/mvtec_ad")
    parser.add_argument("--output", default=None, help="Optional JSON report path")
    args = parser.parse_args()

    result = validate_dataset(args.dataset, args.data_dir)
    encoded = json.dumps(result, indent=2)
    print(encoded)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
