"""VisionGuard CLI training and evaluation runner with full reproducibility metadata logging."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.data.adapters import DATASET_ADAPTERS, get_dataset_adapter
from src.evaluation.metrics import compute_evaluation_metrics
from src.models.registry import MODEL_REGISTRY, build_model


def load_rgb_resized(path: str, size: int) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(im, dtype=np.float32) / 255.0


def load_mask_nearest(path: str | None, size: int) -> np.ndarray:
    from PIL import Image

    if not path or not Path(path).exists():
        return np.zeros((size, size), dtype=np.uint8)
    with Image.open(path) as mk:
        mk = mk.convert("L").resize((size, size), Image.Resampling.NEAREST)
        return (np.asarray(mk, dtype=np.uint8) >= 127).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description="VisionGuard Model Training & Evaluation CLI")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), required=True)
    parser.add_argument("--dataset", choices=list(DATASET_ADAPTERS.keys()), default="mvtec_ad")
    parser.add_argument("--data-dir", default="data/raw/mvtec_ad")
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--output-dir", default="experiments/results")
    args = parser.parse_args()

    adapter = get_dataset_adapter(args.dataset, args.data_dir)
    splits = adapter.load_category(args.category, val_ratio=0.2, seed=args.seed)
    if not splits["train"]:
        print(f"ERROR: No normal training images found for {args.dataset}/{args.category} in {args.data_dir}.")
        return 1

    n_keep = max(2, int(round(len(splits["train"]) * args.train_fraction)))
    train_imgs = [load_rgb_resized(r.image_path, args.image_size) for r in splits["train"][:n_keep]]
    val_imgs = [load_rgb_resized(r.image_path, args.image_size) for r in splits["val"]]

    model = build_model(args.model, seed=args.seed)
    fit_summary = model.fit(train_imgs, val_imgs)

    ckpt_ext = ".pt" if args.model == "autoencoder" else ".npz"
    ckpt_path = Path("models") / args.dataset / args.category / f"{args.model}_seed{args.seed}{ckpt_ext}"
    model.save(ckpt_path)

    labels, scores, masks, maps, latencies = [], [], [], [], []
    for rec in splits["test"]:
        im = load_rgb_resized(rec.image_path, args.image_size)
        t0 = time.perf_counter()
        pred = model.predict(im)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        labels.append(rec.label)
        scores.append(pred.image_score)
        masks.append(load_mask_nearest(rec.mask_path, args.image_size))
        maps.append(pred.anomaly_map)

    threshold = float(fit_summary["threshold_info"]["threshold"])
    metrics = compute_evaluation_metrics(labels, scores, threshold, masks, maps)

    record = {
        "experiment_id": f"{args.dataset}_{args.category}_{args.model}_{int(time.time())}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "category": args.category,
        "model": args.model,
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "train_samples": len(train_imgs),
        "val_samples": len(val_imgs),
        "test_samples": len(splits["test"]),
        "hardware": {"platform": platform.platform(), "python": platform.python_version()},
        "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "checkpoint_path": str(ckpt_path),
        "fit_summary": fit_summary,
        "metrics": metrics,
        "status": "completed",
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{record['experiment_id']}.json"
    out_file.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
