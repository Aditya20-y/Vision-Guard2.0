"""Unified Python model implementations: ConvAutoencoder, PaDiM, PatchCore, FastFlow + Model Registry."""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PredictionBundle:
    image_score: float
    anomaly_map: np.ndarray  # H x W float32
    reconstruction: Optional[np.ndarray] = None  # H x W x 3 float32 in [0, 1]


class BaseAnomalyModel(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, train_images: List[np.ndarray], val_images: List[np.ndarray]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def predict(self, image: np.ndarray) -> PredictionBundle:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str | Path) -> Path:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "BaseAnomalyModel":
        raise NotImplementedError

    @staticmethod
    def calibrate_threshold(val_scores: List[float]) -> Dict[str, float]:
        arr = np.asarray(val_scores, dtype=np.float64)
        mean = float(np.mean(arr)) if len(arr) else 0.0
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        threshold = mean + 2.5 * max(std, mean * 0.04)
        return {
            "policy": "val_normal_mean_plus_2_5_std",
            "threshold": float(threshold),
            "val_mean": mean,
            "val_std": std,
        }


class ConvAutoencoderModel(BaseAnomalyModel):
    """PyTorch Convolutional Autoencoder trained strictly on normal images."""

    name = "autoencoder"

    def __init__(
        self,
        image_size: int = 256,
        latent_dim: int = 128,
        epochs: int = 25,
        lr: float = 1e-3,
        seed: int = 42,
        device: Optional[str] = None,
    ) -> None:
        import torch
        import torch.nn as nn

        self.image_size = image_size
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        torch.manual_seed(seed)

        self.net = nn.Sequential(
            # Encoder
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, latent_dim, 4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            # Decoder
            nn.ConvTranspose2d(latent_dim, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        ).to(self.device)
        self.threshold_info: Dict[str, Any] = {}

    def _to_tensor(self, img: np.ndarray):
        import torch

        t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        return t.to(self.device)

    def fit(self, train_images: List[np.ndarray], val_images: List[np.ndarray]) -> Dict[str, Any]:
        import torch
        import torch.nn.functional as F

        t0 = time.perf_counter()
        self.net.train()
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        xs = torch.cat([self._to_tensor(im) for im in train_images], dim=0)
        loss_history: List[float] = []

        for _ in range(self.epochs):
            opt.zero_grad()
            recon = self.net(xs)
            loss = F.mse_loss(recon, xs)
            loss.backward()
            opt.step()
            loss_history.append(float(loss.item()))

        self.net.eval()
        val_scores = [self.predict(im).image_score for im in (val_images or train_images[:4])]
        self.threshold_info = self.calibrate_threshold(val_scores)
        return {
            "model": self.name,
            "duration_sec": time.perf_counter() - t0,
            "final_train_mse": loss_history[-1] if loss_history else 0.0,
            "threshold_info": self.threshold_info,
        }

    def predict(self, image: np.ndarray) -> PredictionBundle:
        import torch

        self.net.eval()
        with torch.no_grad():
            x = self._to_tensor(image)
            recon = self.net(x)
            err = torch.sqrt(torch.mean((x - recon) ** 2, dim=1)).squeeze(0).cpu().numpy()
            recon_np = recon.squeeze(0).permute(1, 2, 0).cpu().numpy()
        flat = np.sort(err.ravel())
        top_k = max(4, int(len(flat) * 0.015))
        score = float(np.mean(flat[-top_k:]))
        return PredictionBundle(image_score=score, anomaly_map=err.astype(np.float32), reconstruction=recon_np)

    def save(self, path: str | Path) -> Path:
        import torch

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "arch": self.name,
                "image_size": self.image_size,
                "latent_dim": self.latent_dim,
                "state_dict": self.net.state_dict(),
                "threshold_info": self.threshold_info,
            },
            out,
        )
        return out

    @classmethod
    def load(cls, path: str | Path) -> "ConvAutoencoderModel":
        import torch

        ckpt = torch.load(path, map_location="cpu")
        model = cls(image_size=ckpt["image_size"], latent_dim=ckpt["latent_dim"])
        model.net.load_state_dict(ckpt["state_dict"])
        model.threshold_info = ckpt.get("threshold_info", {})
        return model


class _WideResNetBackbone:
    """Frozen pretrained WideResNet-50-2 intermediate feature extractor for PaDiM & PatchCore."""

    def __init__(self, layers: Tuple[str, ...] = ("layer2", "layer3"), device: Optional[str] = None):
        import torch
        from torchvision import models

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.layers = layers
        weights = models.Wide_ResNet50_2_Weights.DEFAULT
        backbone = models.wide_resnet50_2(weights=weights).to(self.device).eval()
        self.backbone = backbone
        self._outputs: Dict[str, Any] = {}
        for name in layers:
            getattr(self.backbone, name).register_forward_hook(self._make_hook(name))

    def _make_hook(self, name: str):
        def _hook(_module, _inp, out):
            self._outputs[name] = out.detach()

        return _hook

    def extract(self, image: np.ndarray):
        import torch
        import torch.nn.functional as F

        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        x = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(self.device)
        x = (x - mean) / std

        self._outputs.clear()
        with torch.no_grad():
            _ = self.backbone(x)

        feats = [self._outputs[k] for k in self.layers]
        target_hw = feats[0].shape[-2:]
        aligned = [
            F.interpolate(f, size=target_hw, mode="bilinear", align_corners=False) for f in feats
        ]
        return torch.cat(aligned, dim=1).squeeze(0)  # [C, H_f, W_f]


class PaDiMModel(BaseAnomalyModel):
    """PaDiM: Per-patch multivariate Gaussian distribution + Mahalanobis anomaly scoring."""

    name = "padim"

    def __init__(self, reduced_dim: int = 100, eps: float = 0.01, seed: int = 42) -> None:
        self.reduced_dim = reduced_dim
        self.eps = eps
        self.seed = seed
        self.backbone: Optional[_WideResNetBackbone] = None
        self.selected_idx: Optional[np.ndarray] = None
        self.means: Optional[np.ndarray] = None
        self.inv_covs: Optional[np.ndarray] = None
        self.threshold_info: Dict[str, Any] = {}

    def _ensure_backbone(self) -> _WideResNetBackbone:
        if self.backbone is None:
            self.backbone = _WideResNetBackbone(layers=("layer1", "layer2", "layer3"))
        return self.backbone

    def fit(self, train_images: List[np.ndarray], val_images: List[np.ndarray]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        bb = self._ensure_backbone()
        rng = np.random.default_rng(self.seed)
        embeddings = [bb.extract(im).cpu().numpy() for im in train_images]
        arr = np.stack(embeddings, axis=0)  # [N, C, H, W]
        n, c, h, w = arr.shape

        dim = min(self.reduced_dim, c)
        self.selected_idx = np.sort(rng.choice(c, size=dim, replace=False))
        sub = arr[:, self.selected_idx, :, :].reshape(n, dim, h * w)

        self.means = np.mean(sub, axis=0)  # [dim, HW]
        self.inv_covs = np.zeros((h * w, dim, dim), dtype=np.float32)
        eye = np.eye(dim, dtype=np.float64) * self.eps
        for p in range(h * w):
            cov = np.cov(sub[:, :, p], rowvar=False) + eye
            self.inv_covs[p] = np.linalg.inv(cov).astype(np.float32)

        val_scores = [self.predict(im).image_score for im in (val_images or train_images[:4])]
        self.threshold_info = self.calibrate_threshold(val_scores)
        return {
            "model": self.name,
            "duration_sec": time.perf_counter() - t0,
            "feature_dim": dim,
            "threshold_info": self.threshold_info,
        }

    def predict(self, image: np.ndarray) -> PredictionBundle:
        import cv2

        assert self.selected_idx is not None and self.means is not None and self.inv_covs is not None
        bb = self._ensure_backbone()
        feat = bb.extract(image).cpu().numpy()[self.selected_idx]  # [dim, H, W]
        dim, h, w = feat.shape
        flat = feat.reshape(dim, h * w)

        dists = np.zeros(h * w, dtype=np.float32)
        for p in range(h * w):
            diff = flat[:, p] - self.means[:, p]
            dists[p] = float(np.sqrt(max(0.0, diff @ self.inv_covs[p] @ diff)))
        grid = dists.reshape(h, w)
        upsampled = cv2.resize(grid, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        smoothed = cv2.GaussianBlur(upsampled, (0, 0), sigmaX=4.0)
        return PredictionBundle(image_score=float(np.max(smoothed)), anomaly_map=smoothed.astype(np.float32))

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            selected_idx=self.selected_idx,
            means=self.means,
            inv_covs=self.inv_covs,
            threshold_info=json.dumps(self.threshold_info),
        )
        return out

    @classmethod
    def load(cls, path: str | Path) -> "PaDiMModel":
        data = np.load(path, allow_pickle=False)
        model = cls()
        model.selected_idx = data["selected_idx"]
        model.means = data["means"]
        model.inv_covs = data["inv_covs"]
        model.threshold_info = json.loads(str(data["threshold_info"]))
        return model


class PatchCoreModel(BaseAnomalyModel):
    """PatchCore: Locally-aware patch embeddings + greedy k-center coreset + k-NN scoring."""

    name = "patchcore"

    def __init__(self, coreset_ratio: float = 0.1, k_neighbors: int = 9, seed: int = 42) -> None:
        self.coreset_ratio = coreset_ratio
        self.k_neighbors = k_neighbors
        self.seed = seed
        self.backbone: Optional[_WideResNetBackbone] = None
        self.memory_bank: Optional[np.ndarray] = None
        self.threshold_info: Dict[str, Any] = {}

    def _ensure_backbone(self) -> _WideResNetBackbone:
        if self.backbone is None:
            self.backbone = _WideResNetBackbone(layers=("layer2", "layer3"))
        return self.backbone

    @staticmethod
    def _greedy_coreset(vectors: np.ndarray, ratio: float, seed: int) -> np.ndarray:
        n = len(vectors)
        target = max(16, min(n, int(round(n * ratio))))
        rng = np.random.default_rng(seed)
        selected = [int(rng.integers(0, n))]
        min_dist = np.full(n, np.inf, dtype=np.float32)

        for _ in range(1, target):
            last_vec = vectors[selected[-1]]
            d = np.linalg.norm(vectors - last_vec, axis=1)
            min_dist = np.minimum(min_dist, d)
            selected.append(int(np.argmax(min_dist)))
        return vectors[selected]

    def fit(self, train_images: List[np.ndarray], val_images: List[np.ndarray]) -> Dict[str, Any]:
        import torch.nn.functional as F

        t0 = time.perf_counter()
        bb = self._ensure_backbone()
        patches: List[np.ndarray] = []
        for im in train_images:
            feat = bb.extract(im).unsqueeze(0)
            pooled = F.avg_pool2d(feat, kernel_size=3, stride=1, padding=1).squeeze(0)
            c, h, w = pooled.shape
            patches.append(pooled.permute(1, 2, 0).reshape(h * w, c).cpu().numpy())

        all_vecs = np.concatenate(patches, axis=0).astype(np.float32)
        self.memory_bank = self._greedy_coreset(all_vecs, self.coreset_ratio, self.seed)

        val_scores = [self.predict(im).image_score for im in (val_images or train_images[:4])]
        self.threshold_info = self.calibrate_threshold(val_scores)
        return {
            "model": self.name,
            "duration_sec": time.perf_counter() - t0,
            "memory_bank_vectors": int(len(self.memory_bank)),
            "threshold_info": self.threshold_info,
        }

    def predict(self, image: np.ndarray) -> PredictionBundle:
        import cv2
        import torch.nn.functional as F

        assert self.memory_bank is not None
        bb = self._ensure_backbone()
        feat = bb.extract(image).unsqueeze(0)
        pooled = F.avg_pool2d(feat, kernel_size=3, stride=1, padding=1).squeeze(0)
        c, h, w = pooled.shape
        q = pooled.permute(1, 2, 0).reshape(h * w, c).cpu().numpy()

        # Exact Euclidean distance matrix to coreset memory bank
        dists = np.linalg.norm(q[:, None, :] - self.memory_bank[None, :, :], axis=2)
        k = min(self.k_neighbors, self.memory_bank.shape[0])
        nearest = np.partition(dists, k - 1, axis=1)[:, :k]
        patch_scores = 0.65 * np.min(nearest, axis=1) + 0.35 * np.mean(nearest, axis=1)

        grid = patch_scores.reshape(h, w).astype(np.float32)
        upsampled = cv2.resize(grid, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        smoothed = cv2.GaussianBlur(upsampled, (0, 0), sigmaX=4.0)
        return PredictionBundle(image_score=float(np.max(smoothed)), anomaly_map=smoothed.astype(np.float32))

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            memory_bank=self.memory_bank,
            k_neighbors=self.k_neighbors,
            threshold_info=json.dumps(self.threshold_info),
        )
        return out

    @classmethod
    def load(cls, path: str | Path) -> "PatchCoreModel":
        data = np.load(path, allow_pickle=False)
        model = cls(k_neighbors=int(data["k_neighbors"]))
        model.memory_bank = data["memory_bank"]
        model.threshold_info = json.loads(str(data["threshold_info"]))
        return model


MODEL_REGISTRY = {
    "autoencoder": ConvAutoencoderModel,
    "padim": PaDiMModel,
    "patchcore": PatchCoreModel,
}


def build_model(model_name: str, **kwargs: Any) -> BaseAnomalyModel:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_name}' is not registered. Available: {list(MODEL_REGISTRY)}. "
            "VisionGuard never silently substitutes an unsupported model."
        )
    return MODEL_REGISTRY[model_name](**kwargs)
