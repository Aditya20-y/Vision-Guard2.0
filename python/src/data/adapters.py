"""Unified dataset interface and adapters for MVTec AD, VisA, MVTec LOCO AD, MVTec AD 2, and BTAD."""
from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class SampleRecord:
    dataset: str
    category: str
    split: str  # train | val | test
    image_path: str
    label: int  # 0 = normal, 1 = anomalous
    defect_type: str
    anomaly_kind: str = "normal"  # normal | structural | logical | shift
    mask_path: Optional[str] = None


@dataclass
class DatasetMeta:
    dataset_id: str
    name: str
    official_url: str
    license: str
    citation: str
    optional: bool = False


class BaseAnomalyDatasetAdapter(ABC):
    """Consistent interface exposed by every independent dataset adapter."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)

    @property
    @abstractmethod
    def metadata(self) -> DatasetMeta:
        raise NotImplementedError

    @abstractmethod
    def list_categories(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def load_category(
        self, category: str, val_ratio: float = 0.2, seed: int = 42
    ) -> Dict[str, List[SampleRecord]]:
        """Returns {'train': [...], 'val': [...], 'test': [...]} with zero test leakage."""
        raise NotImplementedError

    def _split_normal_train_val(
        self, normals: List[SampleRecord], val_ratio: float, seed: int
    ) -> Tuple[List[SampleRecord], List[SampleRecord]]:
        import random

        if len(normals) < 5 or val_ratio <= 0:
            return normals, []
        rng = random.Random(seed)
        shuffled = list(normals)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_ratio)))
        val_samples = [
            SampleRecord(**{**asdict(s), "split": "val"}) for s in shuffled[:n_val]
        ]
        train_samples = shuffled[n_val:]
        return train_samples, val_samples


def _iter_images(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


class MVTecADAdapter(BaseAnomalyDatasetAdapter):
    @property
    def metadata(self) -> DatasetMeta:
        return DatasetMeta(
            dataset_id="mvtec_ad",
            name="MVTec AD",
            official_url="https://www.mvtec.com/research-teaching/datasets/mvtec-ad",
            license="CC BY-NC-SA 4.0",
            citation="Bergmann et al., CVPR 2019",
        )

    def list_categories(self) -> List[str]:
        if not self.root_dir.exists():
            return []
        return sorted(
            d.name for d in self.root_dir.iterdir() if d.is_dir() and (d / "train").exists()
        )

    def load_category(
        self, category: str, val_ratio: float = 0.2, seed: int = 42
    ) -> Dict[str, List[SampleRecord]]:
        cat_dir = self.root_dir / category
        train_normals = [
            SampleRecord("mvtec_ad", category, "train", str(p), 0, "good", "normal")
            for p in _iter_images(cat_dir / "train" / "good")
        ]
        train_split, val_split = self._split_normal_train_val(train_normals, val_ratio, seed)

        test_records: List[SampleRecord] = []
        test_root = cat_dir / "test"
        gt_root = cat_dir / "ground_truth"
        if test_root.exists():
            for defect_dir in sorted(d for d in test_root.iterdir() if d.is_dir()):
                is_good = defect_dir.name == "good"
                for img_path in _iter_images(defect_dir):
                    mask_path: Optional[str] = None
                    if not is_good:
                        candidate = gt_root / defect_dir.name / f"{img_path.stem}_mask.png"
                        if candidate.exists():
                            mask_path = str(candidate)
                    test_records.append(
                        SampleRecord(
                            dataset="mvtec_ad",
                            category=category,
                            split="test",
                            image_path=str(img_path),
                            label=0 if is_good else 1,
                            defect_type=defect_dir.name,
                            anomaly_kind="normal" if is_good else "structural",
                            mask_path=mask_path,
                        )
                    )
        return {"train": train_split, "val": val_split, "test": test_records}


class VisAAdapter(BaseAnomalyDatasetAdapter):
    """Dedicated VisA adapter supporting official split_csv/1cls.csv and Data/Images/Masks layout."""

    @property
    def metadata(self) -> DatasetMeta:
        return DatasetMeta(
            dataset_id="visa",
            name="VisA",
            official_url="https://registry.opendata.aws/visa/",
            license="CC BY 4.0",
            citation="Zou et al., ECCV 2022 (SPot-the-Difference)",
        )

    def list_categories(self) -> List[str]:
        if not self.root_dir.exists():
            return []
        return sorted(
            d.name
            for d in self.root_dir.iterdir()
            if d.is_dir() and (d / "Data" / "Images").exists()
        )

    def load_category(
        self, category: str, val_ratio: float = 0.2, seed: int = 42
    ) -> Dict[str, List[SampleRecord]]:
        split_csv = self.root_dir / "split_csv" / "1cls.csv"
        train_normals: List[SampleRecord] = []
        test_records: List[SampleRecord] = []

        if split_csv.exists():
            with split_csv.open("r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    if row.get("object") != category:
                        continue
                    split = row.get("split", "test").strip().lower()
                    is_anom = row.get("label", "normal").strip().lower() != "normal"
                    img_rel = row.get("image", "").strip()
                    mask_rel = row.get("mask", "").strip()
                    rec = SampleRecord(
                        dataset="visa",
                        category=category,
                        split="train" if split == "train" else "test",
                        image_path=str(self.root_dir / img_rel),
                        label=1 if is_anom else 0,
                        defect_type="anomaly" if is_anom else "normal",
                        anomaly_kind="structural" if is_anom else "normal",
                        mask_path=str(self.root_dir / mask_rel) if mask_rel else None,
                    )
                    if split == "train" and not is_anom:
                        train_normals.append(rec)
                    else:
                        test_records.append(rec)
        else:
            # Fallback to official Data/Images/{Normal,Anomaly} layout
            cat_data = self.root_dir / category / "Data"
            normals = _iter_images(cat_data / "Images" / "Normal")
            split_idx = int(len(normals) * 0.8)
            for p in normals[:split_idx]:
                train_normals.append(
                    SampleRecord("visa", category, "train", str(p), 0, "normal", "normal")
                )
            for p in normals[split_idx:]:
                test_records.append(
                    SampleRecord("visa", category, "test", str(p), 0, "normal", "normal")
                )
            for p in _iter_images(cat_data / "Images" / "Anomaly"):
                mask_p = cat_data / "Masks" / "Anomaly" / f"{p.stem}.png"
                test_records.append(
                    SampleRecord(
                        "visa",
                        category,
                        "test",
                        str(p),
                        1,
                        "anomaly",
                        "structural",
                        str(mask_p) if mask_p.exists() else None,
                    )
                )

        train_split, val_split = self._split_normal_train_val(train_normals, val_ratio, seed)
        return {"train": train_split, "val": val_split, "test": test_records}


class MVTecLOCOAdapter(BaseAnomalyDatasetAdapter):
    """MVTec LOCO AD adapter preserving official validation split and structural vs logical labels."""

    @property
    def metadata(self) -> DatasetMeta:
        return DatasetMeta(
            dataset_id="mvtec_loco",
            name="MVTec LOCO AD",
            official_url="https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad",
            license="CC BY-NC-SA 4.0",
            citation="Bergmann et al., IJCV 2022",
        )

    def list_categories(self) -> List[str]:
        if not self.root_dir.exists():
            return []
        return sorted(
            d.name for d in self.root_dir.iterdir() if d.is_dir() and (d / "train").exists()
        )

    def load_category(
        self, category: str, val_ratio: float = 0.2, seed: int = 42
    ) -> Dict[str, List[SampleRecord]]:
        cat_dir = self.root_dir / category
        train_split = [
            SampleRecord("mvtec_loco", category, "train", str(p), 0, "good", "normal")
            for p in _iter_images(cat_dir / "train" / "good")
        ]
        val_split = [
            SampleRecord("mvtec_loco", category, "val", str(p), 0, "good", "normal")
            for p in _iter_images(cat_dir / "validation" / "good")
        ]
        if not val_split:
            train_split, val_split = self._split_normal_train_val(train_split, val_ratio, seed)

        test_records: List[SampleRecord] = []
        for sub, label, kind in [
            ("good", 0, "normal"),
            ("structural_anomalies", 1, "structural"),
            ("logical_anomalies", 1, "logical"),
        ]:
            for img_path in _iter_images(cat_dir / "test" / sub):
                mask_dir = cat_dir / "ground_truth" / sub / img_path.stem
                mask_file = next(iter(_iter_images(mask_dir)), None) if mask_dir.exists() else None
                test_records.append(
                    SampleRecord(
                        dataset="mvtec_loco",
                        category=category,
                        split="test",
                        image_path=str(img_path),
                        label=label,
                        defect_type=sub,
                        anomaly_kind=kind,
                        mask_path=str(mask_file) if mask_file else None,
                    )
                )
        return {"train": train_split, "val": val_split, "test": test_records}


class MVTecAD2Adapter(BaseAnomalyDatasetAdapter):
    """MVTec AD 2 adapter strictly restricted to official public evaluation splits."""

    @property
    def metadata(self) -> DatasetMeta:
        return DatasetMeta(
            dataset_id="mvtec_ad2",
            name="MVTec AD 2",
            official_url="https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2",
            license="CC BY-NC-SA 4.0",
            citation="Heckler et al., 2025",
        )

    def list_categories(self) -> List[str]:
        if not self.root_dir.exists():
            return []
        return sorted(
            d.name for d in self.root_dir.iterdir() if d.is_dir() and (d / "train").exists()
        )

    def load_category(
        self, category: str, val_ratio: float = 0.2, seed: int = 42
    ) -> Dict[str, List[SampleRecord]]:
        cat_dir = self.root_dir / category
        train_split = [
            SampleRecord("mvtec_ad2", category, "train", str(p), 0, "good", "normal")
            for p in _iter_images(cat_dir / "train" / "good")
        ]
        val_split = [
            SampleRecord("mvtec_ad2", category, "val", str(p), 0, "good", "normal")
            for p in _iter_images(cat_dir / "validation" / "good")
        ]
        if not val_split:
            train_split, val_split = self._split_normal_train_val(train_split, val_ratio, seed)

        test_records: List[SampleRecord] = []
        pub_root = cat_dir / "test_public"
        for sub, label in [("good", 0), ("bad", 1)]:
            for img_path in _iter_images(pub_root / sub):
                mask_p = pub_root / "ground_truth" / "bad" / f"{img_path.stem}_mask.png"
                test_records.append(
                    SampleRecord(
                        dataset="mvtec_ad2",
                        category=category,
                        split="test",
                        image_path=str(img_path),
                        label=label,
                        defect_type=sub,
                        anomaly_kind="shift" if label == 1 else "normal",
                        mask_path=str(mask_p) if (label == 1 and mask_p.exists()) else None,
                    )
                )
        return {"train": train_split, "val": val_split, "test": test_records}


class BTADAdapter(BaseAnomalyDatasetAdapter):
    """Optional BTAD adapter (does not block core evaluation if dataset is unavailable)."""

    @property
    def metadata(self) -> DatasetMeta:
        return DatasetMeta(
            dataset_id="btad",
            name="BTAD (BeanTech Anomaly Detection)",
            official_url="http://avires.dimi.uniud.it/papers/btad/btad.zip",
            license="Academic / Research (verify AVIRES Lab terms)",
            citation="Mishra et al., ISIE 2021",
            optional=True,
        )

    def list_categories(self) -> List[str]:
        if not self.root_dir.exists():
            return []
        return sorted(
            d.name for d in self.root_dir.iterdir() if d.is_dir() and (d / "train").exists()
        )

    def load_category(
        self, category: str, val_ratio: float = 0.2, seed: int = 42
    ) -> Dict[str, List[SampleRecord]]:
        cat_dir = self.root_dir / category
        train_normals = [
            SampleRecord("btad", category, "train", str(p), 0, "ok", "normal")
            for p in _iter_images(cat_dir / "train" / "ok")
        ]
        train_split, val_split = self._split_normal_train_val(train_normals, val_ratio, seed)

        test_records: List[SampleRecord] = []
        for sub, label in [("ok", 0), ("ko", 1)]:
            for img_path in _iter_images(cat_dir / "test" / sub):
                mask_p = cat_dir / "ground_truth" / "ko" / f"{img_path.stem}.png"
                test_records.append(
                    SampleRecord(
                        dataset="btad",
                        category=category,
                        split="test",
                        image_path=str(img_path),
                        label=label,
                        defect_type=sub,
                        anomaly_kind="structural" if label == 1 else "normal",
                        mask_path=str(mask_p) if (label == 1 and mask_p.exists()) else None,
                    )
                )
        return {"train": train_split, "val": val_split, "test": test_records}


DATASET_ADAPTERS = {
    "mvtec_ad": MVTecADAdapter,
    "visa": VisAAdapter,
    "mvtec_loco": MVTecLOCOAdapter,
    "mvtec_ad2": MVTecAD2Adapter,
    "btad": BTADAdapter,
}


def get_dataset_adapter(dataset_id: str, root_dir: str | Path) -> BaseAnomalyDatasetAdapter:
    if dataset_id not in DATASET_ADAPTERS:
        raise ValueError(f"Unsupported dataset '{dataset_id}'. Available: {list(DATASET_ADAPTERS)}")
    return DATASET_ADAPTERS[dataset_id](root_dir)
