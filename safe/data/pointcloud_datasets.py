"""
Point cloud datasets for SAFE architecture.

Mirrors the interface of AudioCapsDataset for modality-agnostic training.
"""

from __future__ import annotations

import json
import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# Point Cloud Augmentation Functions
# Based on PointNeXt (NeurIPS 2022) best practices
# =============================================================================

def random_rotate_pointcloud(pc: np.ndarray, rotation_range: float = 180.0) -> np.ndarray:
    """
    Randomly rotate point cloud around the Y-axis (up axis).

    Args:
        pc: Point cloud of shape (N, 3) or (N, 6) with normals
        rotation_range: Max rotation in degrees (default: full 360 via ±180)

    Returns:
        Rotated point cloud
    """
    theta = np.random.uniform(-rotation_range, rotation_range) * np.pi / 180.0
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    # Rotation matrix around Y-axis
    rotation_matrix = np.array([
        [cos_t, 0, sin_t],
        [0, 1, 0],
        [-sin_t, 0, cos_t]
    ])

    pc_rotated = pc.copy()
    pc_rotated[:, :3] = pc[:, :3] @ rotation_matrix.T

    # Rotate normals if present
    if pc.shape[1] > 3:
        pc_rotated[:, 3:6] = pc[:, 3:6] @ rotation_matrix.T

    return pc_rotated


def random_scale_pointcloud(pc: np.ndarray, scale_range: tuple = (0.8, 1.2)) -> np.ndarray:
    """
    Randomly scale point cloud uniformly.

    Args:
        pc: Point cloud of shape (N, 3+)
        scale_range: (min_scale, max_scale) tuple

    Returns:
        Scaled point cloud
    """
    scale = np.random.uniform(scale_range[0], scale_range[1])
    pc_scaled = pc.copy()
    pc_scaled[:, :3] = pc[:, :3] * scale
    return pc_scaled


def random_jitter_pointcloud(pc: np.ndarray, sigma: float = 0.01, clip: float = 0.05) -> np.ndarray:
    """
    Add random Gaussian noise to point positions.

    Args:
        pc: Point cloud of shape (N, 3+)
        sigma: Standard deviation of Gaussian noise
        clip: Maximum absolute noise value

    Returns:
        Jittered point cloud
    """
    noise = np.clip(np.random.normal(0, sigma, (pc.shape[0], 3)), -clip, clip)
    pc_jittered = pc.copy()
    pc_jittered[:, :3] = pc[:, :3] + noise
    return pc_jittered


def random_translate_pointcloud(pc: np.ndarray, translate_range: float = 0.2) -> np.ndarray:
    """
    Randomly translate point cloud.

    Args:
        pc: Point cloud of shape (N, 3+)
        translate_range: Maximum translation in each axis

    Returns:
        Translated point cloud
    """
    translation = np.random.uniform(-translate_range, translate_range, 3)
    pc_translated = pc.copy()
    pc_translated[:, :3] = pc[:, :3] + translation
    return pc_translated


def random_dropout_pointcloud(pc: np.ndarray, max_dropout_ratio: float = 0.875) -> np.ndarray:
    """
    Randomly drop points from point cloud.

    Args:
        pc: Point cloud of shape (N, 3+)
        max_dropout_ratio: Maximum fraction of points to drop

    Returns:
        Point cloud with some points dropped (and duplicated to maintain size)
    """
    dropout_ratio = np.random.uniform(0, max_dropout_ratio)
    n_points = pc.shape[0]
    n_drop = int(n_points * dropout_ratio)

    if n_drop == 0:
        return pc

    # Randomly select points to keep
    keep_indices = np.random.choice(n_points, n_points - n_drop, replace=False)
    pc_dropped = pc[keep_indices]

    # Duplicate random points to maintain original size
    if len(pc_dropped) < n_points:
        dup_indices = np.random.choice(len(pc_dropped), n_points - len(pc_dropped), replace=True)
        pc_dropped = np.concatenate([pc_dropped, pc_dropped[dup_indices]], axis=0)

    return pc_dropped


def augment_pointcloud(
    pc: np.ndarray,
    rotate: bool = True,
    scale: bool = True,
    jitter: bool = True,
    translate: bool = True,
    dropout: bool = False,
    rotation_range: float = 180.0,
    scale_range: tuple = (0.8, 1.2),
    jitter_sigma: float = 0.01,
    jitter_clip: float = 0.05,
    translate_range: float = 0.2,
    dropout_ratio: float = 0.875,
) -> np.ndarray:
    """
    Apply a sequence of augmentations to a point cloud.

    Based on PointNeXt best practices for ModelNet40.

    Args:
        pc: Point cloud of shape (N, 3+)
        rotate: Apply random Y-axis rotation
        scale: Apply random uniform scaling
        jitter: Apply random Gaussian noise
        translate: Apply random translation
        dropout: Apply random point dropout
        *_range/*_sigma: Parameters for each augmentation

    Returns:
        Augmented point cloud
    """
    if rotate:
        pc = random_rotate_pointcloud(pc, rotation_range)
    if scale:
        pc = random_scale_pointcloud(pc, scale_range)
    if translate:
        pc = random_translate_pointcloud(pc, translate_range)
    if jitter:
        pc = random_jitter_pointcloud(pc, jitter_sigma, jitter_clip)
    if dropout:
        pc = random_dropout_pointcloud(pc, dropout_ratio)

    return pc


# ModelNet40 class names (alphabetical order)
MODELNET40_CLASSES = [
    "airplane", "bathtub", "bed", "bench", "bookshelf",
    "bottle", "bowl", "car", "chair", "cone",
    "cup", "curtain", "desk", "door", "dresser",
    "flower_pot", "glass_box", "guitar", "keyboard", "lamp",
    "laptop", "mantel", "monitor", "night_stand", "person",
    "piano", "plant", "radio", "range_hood", "sink",
    "sofa", "stairs", "stool", "table", "tent",
    "toilet", "tv_stand", "vase", "wardrobe", "xbox",
]


class ModelNet40Dataset(Dataset):
    """
    ModelNet40 3D object classification dataset.

    Supports both:
    - HDF5 format (from antao97/PointCloudDatasets)
    - Original .off mesh files

    Returns samples in SAFE-compatible format for classification as generation.
    """

    dataset_name = "modelnet40"
    num_classes = 40
    class_names = MODELNET40_CLASSES

    def __init__(
        self,
        data_path: str | Path,
        split: str = "train",
        num_points: int = 1024,
        use_normals: bool = False,
        augment: bool = None,  # None = auto (True for train, False for test)
        aug_rotate: bool = True,
        aug_scale: bool = True,
        aug_jitter: bool = True,
        aug_translate: bool = True,
        aug_dropout: bool = False,
    ):
        """
        Initialize ModelNet40 dataset.

        Args:
            data_path: Root data directory containing modelnet40/
            split: "train" or "test"
            num_points: Number of points to sample from each object
            use_normals: Whether to include point normals (6D instead of 3D)
            augment: Whether to apply data augmentation (default: True for train)
            aug_rotate: Apply random rotation around Y-axis
            aug_scale: Apply random scaling [0.8, 1.2]
            aug_jitter: Apply random Gaussian noise (sigma=0.01)
            aug_translate: Apply random translation
            aug_dropout: Apply random point dropout
        """
        self.data_path = Path(data_path)
        self.split = split.lower()
        self.num_points = num_points
        self.use_normals = use_normals

        # Augmentation settings
        self.augment = augment if augment is not None else (self.split == "train")
        self.aug_rotate = aug_rotate
        self.aug_scale = aug_scale
        self.aug_jitter = aug_jitter
        self.aug_translate = aug_translate
        self.aug_dropout = aug_dropout

        # Find dataset directory
        dataset_dir = self.data_path / self.dataset_name
        if not dataset_dir.exists():
            dataset_dir = self.data_path  # Try root directly

        # Try HDF5 format first (faster)
        h5_file = dataset_dir / f"modelnet40_{self.split}.h5"
        if h5_file.exists():
            self._load_h5(h5_file)
        else:
            # Fall back to directory structure
            self._load_from_directory(dataset_dir)

        print(f"[ModelNet40] Loaded {len(self.pointclouds)} samples ({self.split})", flush=True)

    def _load_h5(self, h5_path: Path) -> None:
        """Load from HDF5 file."""
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required for HDF5 format: pip install h5py")

        with h5py.File(h5_path, "r") as f:
            self.pointclouds = f["data"][:]  # (N, num_points, 3 or 6)
            self.labels = f["label"][:].squeeze()  # (N,)

        # Ensure labels are integers
        self.labels = self.labels.astype(np.int64)

    def _load_from_directory(self, dataset_dir: Path) -> None:
        """Load from directory of .npy or .off files."""
        self.pointclouds = []
        self.labels = []

        for class_idx, class_name in enumerate(self.class_names):
            class_dir = dataset_dir / class_name / self.split
            if not class_dir.exists():
                continue

            for pc_file in sorted(class_dir.glob("*.npy")):
                pc = np.load(pc_file)
                self.pointclouds.append(pc[:self.num_points])
                self.labels.append(class_idx)

        self.pointclouds = np.array(self.pointclouds)
        self.labels = np.array(self.labels)

    def __len__(self) -> int:
        return len(self.pointclouds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample in SAFE-compatible format.

        For classification, we frame it as a generation task:
        Q: "What type of object is this?"
        A: "airplane" (class name)
        """
        pc = self.pointclouds[idx].copy()  # Copy to avoid modifying original
        label = self.labels[idx]
        class_name = self.class_names[label]

        # Subsample if needed
        if len(pc) > self.num_points:
            indices = np.random.choice(len(pc), self.num_points, replace=False)
            pc = pc[indices]
        elif len(pc) < self.num_points:
            # Pad by repeating
            pad_size = self.num_points - len(pc)
            pad_indices = np.random.choice(len(pc), pad_size, replace=True)
            pc = np.concatenate([pc, pc[pad_indices]], axis=0)

        # Take only xyz
        if pc.shape[-1] > 3 and not self.use_normals:
            pc = pc[:, :3]

        # Apply augmentations during training
        if self.augment:
            pc = augment_pointcloud(
                pc,
                rotate=self.aug_rotate,
                scale=self.aug_scale,
                jitter=self.aug_jitter,
                translate=self.aug_translate,
                dropout=self.aug_dropout,
            )

        return {
            "sample_id": f"modelnet40_{self.split}_{idx}",
            "question": "What type of object is this?",
            "answers": class_name,
            "pointcloud": torch.from_numpy(pc).float(),
            "label": label,
            "valid": True,
            "images": None,
            "audio": None,
        }


class Cap3DDataset(Dataset):
    """
    Cap3D dataset: 3D object captions from Objaverse.

    Source: https://huggingface.co/datasets/tiange/Cap3D
    ~660K 3D objects with captions.
    """

    dataset_name = "cap3d"

    def __init__(
        self,
        data_path: str | Path,
        split: str = "train",
        num_points: int = 8192,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize Cap3D dataset.

        Args:
            data_path: Root data directory containing cap3d/
            split: "train", "val", or "test"
            num_points: Number of points to sample
            max_samples: Optional limit on number of samples
        """
        self.data_path = Path(data_path)
        self.split = split.lower()
        self.num_points = num_points

        dataset_dir = self.data_path / self.dataset_name
        if not dataset_dir.exists():
            dataset_dir = self.data_path

        # Load captions
        captions_file = dataset_dir / f"cap3d_{self.split}.json"
        if not captions_file.exists():
            captions_file = dataset_dir / "Cap3D_automated_Objaverse.json"

        if captions_file.exists():
            with open(captions_file, "r") as f:
                self.captions = json.load(f)
        else:
            raise FileNotFoundError(f"Cap3D captions not found at {captions_file}")

        # Filter to samples that have point cloud files
        self.pointcloud_dir = dataset_dir / "pointclouds"
        if not self.pointcloud_dir.exists():
            raise FileNotFoundError(
                f"Cap3D pointcloud directory not found at {self.pointcloud_dir}. "
                f"Expected structure like: {dataset_dir}/pointclouds/<object_id>.npy"
            )
        self.examples = []

        for obj_id, caption in self.captions.items():
            pc_path = self.pointcloud_dir / f"{obj_id}.npy"
            if pc_path.exists():
                self.examples.append({
                    "object_id": obj_id,
                    "caption": caption,
                    "pc_path": pc_path,
                })

            if max_samples and len(self.examples) >= max_samples:
                break

        print(f"[Cap3D] Loaded {len(self.examples)} samples ({self.split})", flush=True)
        if len(self.examples) == 0:
            raise RuntimeError(
                f"[Cap3D] Found 0 usable samples. Check captions file and pointclouds directory: {self.pointcloud_dir}"
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample in SAFE-compatible format."""
        entry = self.examples[idx]

        # Load point cloud
        valid = True
        try:
            pc = np.load(entry["pc_path"])
        except Exception:
            pc = np.zeros((self.num_points, 3), dtype=np.float32)
            valid = False

        caption = entry["caption"]
        if isinstance(caption, list):
            # Some caption files store multiple captions per object
            caption = next((c for c in caption if isinstance(c, str) and c.strip()), "")

        # Subsample
        if len(pc) > self.num_points:
            indices = np.random.choice(len(pc), self.num_points, replace=False)
            pc = pc[indices]
        elif len(pc) < self.num_points:
            pad_size = self.num_points - len(pc)
            pad_indices = np.random.choice(len(pc), pad_size, replace=True)
            pc = np.concatenate([pc, pc[pad_indices]], axis=0)

        # Take only xyz
        if pc.shape[-1] > 3:
            pc = pc[:, :3]

        return {
            "sample_id": entry["object_id"],
            "question": "Describe this 3D object.",
            "answers": caption,
            "pointcloud": torch.from_numpy(pc).float(),
            "valid": valid,
            "images": None,
            "audio": None,
        }


class ShapeNetPartDataset(Dataset):
    """
    ShapeNet Part segmentation dataset, adapted for captioning.

    Can be used for classification (category prediction) or
    simple captioning based on category + part structure.
    """

    dataset_name = "shapenet_part"

    # ShapeNet categories
    category_names = [
        "airplane", "bag", "cap", "car", "chair",
        "earphone", "guitar", "knife", "lamp", "laptop",
        "motorbike", "mug", "pistol", "rocket", "skateboard", "table",
    ]

    def __init__(
        self,
        data_path: str | Path,
        split: str = "train",
        num_points: int = 2048,
        task: str = "classification",  # or "captioning"
    ):
        self.data_path = Path(data_path)
        self.split = split.lower()
        self.num_points = num_points
        self.task = task

        dataset_dir = self.data_path / self.dataset_name
        if not dataset_dir.exists():
            dataset_dir = self.data_path

        # Load from HDF5
        h5_file = dataset_dir / f"shapenet_{self.split}.h5"
        if h5_file.exists():
            self._load_h5(h5_file)
        else:
            raise FileNotFoundError(f"ShapeNet data not found at {h5_file}")

        print(f"[ShapeNetPart] Loaded {len(self.pointclouds)} samples ({self.split})", flush=True)

    def _load_h5(self, h5_path: Path) -> None:
        import h5py
        with h5py.File(h5_path, "r") as f:
            self.pointclouds = f["data"][:]
            self.labels = f["label"][:].squeeze().astype(np.int64)

    def __len__(self) -> int:
        return len(self.pointclouds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pc = self.pointclouds[idx]
        label = self.labels[idx]
        category = self.category_names[label]

        # Subsample
        if len(pc) > self.num_points:
            indices = np.random.choice(len(pc), self.num_points, replace=False)
            pc = pc[indices]

        if pc.shape[-1] > 3:
            pc = pc[:, :3]

        if self.task == "classification":
            question = "What type of object is this?"
            answer = category
        else:
            question = "Describe this 3D object."
            answer = f"A 3D model of a {category}."

        return {
            "sample_id": f"shapenet_{self.split}_{idx}",
            "question": question,
            "answers": answer,
            "pointcloud": torch.from_numpy(pc).float(),
            "label": label,
            "valid": True,
            "images": None,
            "audio": None,
        }


def collate_pointcloud_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for point cloud batches.

    Mirrors _collate_multimodal_batch from datasets.py.
    """
    # Optionally filter invalid samples
    if batch and any("valid" in s for s in batch):
        batch = [s for s in batch if bool(s.get("valid", True))]

    if not batch:
        return {
            "questions": [],
            "answers": [],
            "pointclouds": None,
            "sample_ids": [],
            "images": None,
            "audio": None,
            "labels": None,
        }

    # Stack point clouds into tensor
    pointclouds = torch.stack([s["pointcloud"] for s in batch])

    # Collect other fields as lists
    result = {
        "questions": [s["question"] for s in batch],
        "answers": [s["answers"] for s in batch],
        "pointclouds": pointclouds,
        "sample_ids": [s["sample_id"] for s in batch],
        "images": None,
        "audio": None,
    }

    # Include labels if available (for classification)
    if "label" in batch[0]:
        result["labels"] = torch.tensor([s["label"] for s in batch])

    return result


def create_pointcloud_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for point cloud datasets.

    Mirrors create_safe_dataloader from datasets.py.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_pointcloud_batch,
        pin_memory=pin_memory,
        drop_last=False,
    )
