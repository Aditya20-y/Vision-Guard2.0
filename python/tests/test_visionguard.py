"""Unit and integration tests for VisionGuard dataset adapters, metrics, and models (synthetic test fixtures only)."""
from __future__ import annotations

from pathlib import Path

from src.data.adapters import (
    BTADAdapter,
    MVTecAD2Adapter,
    MVTecADAdapter,
    MVTecLOCOAdapter,
    VisAAdapter,
    get_dataset_adapter,
)
from src.data.validation import validate_dataset


def test_dataset_registry_adapters(tmp_path: Path) -> None:
    for ds_id, cls in [
        ("mvtec_ad", MVTecADAdapter),
        ("visa", VisAAdapter),
        ("mvtec_loco", MVTecLOCOAdapter),
        ("mvtec_ad2", MVTecAD2Adapter),
        ("btad", BTADAdapter),
    ]:
        adapter = get_dataset_adapter(ds_id, tmp_path / ds_id)
        assert isinstance(adapter, cls)
        assert adapter.list_categories() == []


def test_mvtec_adapter_split_isolation(tmp_path: Path) -> None:
    # Synthetic directory structure strictly for unit testing split separation
    cat = tmp_path / "bottle"
    (cat / "train" / "good").mkdir(parents=True)
    (cat / "test" / "good").mkdir(parents=True)
    (cat / "test" / "broken_large").mkdir(parents=True)
    (cat / "ground_truth" / "broken_large").mkdir(parents=True)

    for i in range(10):
        (cat / "train" / "good" / f"{i:03d}.png").write_bytes(b"fake")
    for i in range(3):
        (cat / "test" / "good" / f"g{i:03d}.png").write_bytes(b"fake")
    for i in range(4):
        (cat / "test" / "broken_large" / f"b{i:03d}.png").write_bytes(b"fake")
        (cat / "ground_truth" / "broken_large" / f"b{i:03d}_mask.png").write_bytes(b"fake")

    adapter = MVTecADAdapter(tmp_path)
    assert adapter.list_categories() == ["bottle"]
    splits = adapter.load_category("bottle", val_ratio=0.2, seed=42)
    assert len(splits["train"]) == 8
    assert len(splits["val"]) == 2
    assert len(splits["test"]) == 7
    assert all(r.label == 0 for r in splits["train"])
    assert all(r.label == 0 for r in splits["val"])
    assert sum(1 for r in splits["test"] if r.label == 1) == 4
    assert all(r.mask_path is not None for r in splits["test"] if r.label == 1)


def test_validation_cli_reports_missing_dataset_cleanly(tmp_path: Path) -> None:
    report = validate_dataset("mvtec_ad", tmp_path / "empty")
    assert report["available"] is False
    assert report["categories"] == {}
