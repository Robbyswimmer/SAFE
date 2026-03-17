#!/usr/bin/env python3
"""
Download and prepare 3D QA datasets (SQA3D, ScanQA, ScanNet preprocessing).

This script downloads:
1. SQA3D annotations from Zenodo
2. ScanQA annotations from GitHub
3. (Optional) Preprocesses ScanNet point clouds + images

Features:
- Resume capability for interrupted downloads
- Progress tracking and logging
- tqdm progress bars
- Verification and coverage reporting

Usage:
    python scripts/download_3d_qa_data.py --data-dir ./data
    python scripts/download_3d_qa_data.py --data-dir ./data --scannet-root /path/to/scannet
    python scripts/download_3d_qa_data.py --data-dir ./data --sqa3d-only
    python scripts/download_3d_qa_data.py --data-dir ./data --validate-only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------
SQA3D_URLS = [
    # Zenodo API endpoint (current format)
    "https://zenodo.org/api/records/7792397/files/sqa_task.zip/content",
    "https://zenodo.org/api/records/7544818/files/sqa_task.zip/content",
    # Legacy format (kept as fallback)
    "https://zenodo.org/records/7792397/files/sqa_task.zip?download=1",
    "https://zenodo.org/records/7544818/files/sqa_task.zip?download=1",
]

SCANQA_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/ATR-DBI/ScanQA/main/data/qa/ScanQA_v1.0_{split}.json"
)
SCANQA_URL_FALLBACKS = [
    "https://raw.githubusercontent.com/ATR-DBI/ScanQA/master/data/qa/ScanQA_v1.0_{split}.json",
    "https://raw.githubusercontent.com/ATR-DBI/ScanQA/main/data/ScanQA_v1.0_{split}.json",
    "https://raw.githubusercontent.com/ATR-DBI/ScanQA/master/data/ScanQA_v1.0_{split}.json",
]

# Expected SQA3D files after extraction
SQA3D_EXPECTED_FILES = [
    "v1_balanced_questions_train_scannetv2.json",
    "v1_balanced_questions_val_scannetv2.json",
    "v1_balanced_questions_test_scannetv2.json",
    "v1_balanced_sqa_annotations_train_scannetv2.json",
    "v1_balanced_sqa_annotations_val_scannetv2.json",
    "v1_balanced_sqa_annotations_test_scannetv2.json",
    "answer_dict.json",
]

SCANQA_SPLITS = ["train", "val", "test"]


def _setup_logging(log_file: Path) -> logging.Logger:
    """Configure logging to file + stderr."""
    logger = logging.getLogger("download_3d_qa")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    fh = logging.FileHandler(str(log_file))
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


def _download_file(url: str, dest: Path, desc: str = "Downloading") -> bool:
    """Stream-download a file with tqdm progress bar. Returns True on success."""
    try:
        resp = requests.get(url, stream=True, timeout=60, allow_redirects=True)
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")
        if "text/html" in content_type and int(resp.headers.get("content-length", "1")) < 1000:
            return False

        total = int(resp.headers.get("content-length", 0))

        with open(dest, "wb") as f:
            with tqdm(total=total, unit="iB", unit_scale=True, desc=desc) as pbar:
                for chunk in resp.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)
        return True
    except Exception as e:
        logging.getLogger("download_3d_qa").warning(f"  Download error: {e}")
        if dest.exists():
            dest.unlink()
        return False


class ThreeDQADownloader:
    """Download SQA3D + ScanQA annotations and optionally preprocess ScanNet."""

    def __init__(
        self,
        data_dir: str,
        scannet_root: Optional[str] = None,
        frames_dir: Optional[str] = None,
        sqa3d_only: bool = False,
        skip_scannet: bool = False,
        validate_only: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.scannet_root = Path(scannet_root) if scannet_root else None
        self.frames_dir = Path(frames_dir) if frames_dir else None
        self.sqa3d_only = sqa3d_only
        self.skip_scannet = skip_scannet
        self.validate_only = validate_only

        # Output directories
        self.sqa3d_dir = self.data_dir / "sqa3d"
        self.scanqa_dir = self.data_dir / "scanqa"
        self.scannet_dir = self.data_dir / "scannet"
        self.pc_dir = self.scannet_dir / "pointclouds"
        self.img_dir = self.scannet_dir / "images"

        # Logging
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logger = _setup_logging(self.data_dir / "download_3d_qa.log")

        # Progress tracking
        self.progress_file = self.data_dir / "3d_qa_progress.json"
        self.progress = self._load_progress()

    # ------------------------------------------------------------------
    # Progress persistence
    # ------------------------------------------------------------------
    def _load_progress(self) -> dict:
        if self.progress_file.exists():
            try:
                with open(self.progress_file) as f:
                    data = json.load(f)
                self.logger.info(f"Loaded progress from {self.progress_file}")
                return data
            except Exception:
                pass
        return {"completed_steps": [], "scannet_scenes_done": []}

    def _save_progress(self):
        try:
            self.progress["timestamp"] = datetime.now().isoformat()
            with open(self.progress_file, "w") as f:
                json.dump(self.progress, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Could not save progress: {e}")

    # ------------------------------------------------------------------
    # Step 1: SQA3D annotations
    # ------------------------------------------------------------------
    def download_sqa3d(self) -> bool:
        self.logger.info("=" * 60)
        self.logger.info("STEP 1: Download SQA3D annotations")
        self.logger.info("=" * 60)

        self.sqa3d_dir.mkdir(parents=True, exist_ok=True)

        # Check if already present
        existing = [f for f in SQA3D_EXPECTED_FILES if (self.sqa3d_dir / f).exists()]
        if len(existing) == len(SQA3D_EXPECTED_FILES):
            self.logger.info("SQA3D annotations already exist, skipping download.")
            self._log_sqa3d_stats()
            return True

        # Download zip
        zip_path = self.data_dir / "sqa_task.zip"
        downloaded = False

        for url in SQA3D_URLS:
            self.logger.info(f"Trying: {url[:80]}...")
            if _download_file(url, zip_path, "SQA3D annotations"):
                downloaded = True
                break
            self.logger.warning(f"  Failed: {url[:80]}")

        if not downloaded:
            self.logger.error("Could not download SQA3D from any URL.")
            return False

        # Extract
        self.logger.info("Extracting sqa_task.zip...")
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                names = zf.namelist()
                self.logger.info(f"  Archive contains {len(names)} files")

                for member in names:
                    basename = os.path.basename(member)
                    if not basename or basename.startswith("."):
                        continue
                    # Extract matching files to sqa3d_dir
                    if basename.endswith(".json"):
                        target = self.sqa3d_dir / basename
                        with zf.open(member) as src, open(target, "wb") as dst:
                            dst.write(src.read())
                        self.logger.info(f"  Extracted: {basename}")
        except zipfile.BadZipFile as e:
            self.logger.error(f"Bad zip file: {e}")
            zip_path.unlink(missing_ok=True)
            return False

        # Cleanup zip
        zip_path.unlink(missing_ok=True)

        # Verify
        found = [f for f in SQA3D_EXPECTED_FILES if (self.sqa3d_dir / f).exists()]
        if len(found) < 3:
            self.logger.error(f"Only found {len(found)}/{len(SQA3D_EXPECTED_FILES)} expected files")
            return False

        self.logger.info(f"SQA3D: {len(found)}/{len(SQA3D_EXPECTED_FILES)} files extracted")
        self._log_sqa3d_stats()

        self.progress["completed_steps"].append("sqa3d")
        self._save_progress()
        return True

    def _log_sqa3d_stats(self):
        """Log SQA3D statistics: questions per split, unique scenes, answer vocab."""
        for split in ["train", "val", "test"]:
            qfile = self.sqa3d_dir / f"v1_balanced_questions_{split}_scannetv2.json"
            if qfile.exists():
                try:
                    with open(qfile) as f:
                        data = json.load(f)
                    questions = data.get("questions", data) if isinstance(data, dict) else data
                    if isinstance(questions, list):
                        self.logger.info(f"  SQA3D {split}: {len(questions)} questions")
                    elif isinstance(questions, dict):
                        self.logger.info(f"  SQA3D {split}: {len(questions)} entries")
                except Exception:
                    pass

        # Answer dict
        answer_dict_file = self.sqa3d_dir / "answer_dict.json"
        if answer_dict_file.exists():
            try:
                with open(answer_dict_file) as f:
                    ad = json.load(f)
                vocab_size = len(ad) if isinstance(ad, (dict, list)) else 0
                self.logger.info(f"  SQA3D answer vocab size: {vocab_size}")
            except Exception:
                pass

        # Unique scenes from annotations
        scenes = set()
        for split in ["train", "val", "test"]:
            afile = self.sqa3d_dir / f"v1_balanced_sqa_annotations_{split}_scannetv2.json"
            if afile.exists():
                try:
                    with open(afile) as f:
                        data = json.load(f)
                    annots = data.get("annotations", data) if isinstance(data, dict) else data
                    if isinstance(annots, list):
                        for a in annots:
                            sid = a.get("scene_id", "")
                            if sid:
                                scenes.add(sid)
                except Exception:
                    pass
        if scenes:
            self.logger.info(f"  SQA3D unique scenes: {len(scenes)}")

    # ------------------------------------------------------------------
    # Step 2: ScanQA annotations
    # ------------------------------------------------------------------
    def download_scanqa(self) -> bool:
        self.logger.info("=" * 60)
        self.logger.info("STEP 2: Download ScanQA annotations")
        self.logger.info("=" * 60)

        self.scanqa_dir.mkdir(parents=True, exist_ok=True)

        # First, look for annotations already present in subdirectories
        # (e.g. from a previous repo clone into data/scanqa/ScanQA_v1.0/ or data/scanqa/repo/)
        self._find_local_scanqa_annotations()

        required_splits = ["train", "val"]  # test is optional (often not publicly available)
        optional_splits = ["test"]
        have_required = True

        for split in SCANQA_SPLITS:
            filename = f"ScanQA_v1.0_{split}.json"
            dest = self.scanqa_dir / filename

            if dest.exists():
                self.logger.info(f"  {filename} already exists, skipping.")
                continue

            # Try primary URL then fallbacks
            urls = [SCANQA_URL_TEMPLATE.format(split=split)] + [
                u.format(split=split) for u in SCANQA_URL_FALLBACKS
            ]

            downloaded = False
            for url in urls:
                try:
                    self.logger.info(f"  Trying: {url[:80]}...")
                    resp = requests.get(url, timeout=30)
                    resp.raise_for_status()
                    payload = resp.json()

                    with open(dest, "w") as f:
                        json.dump(payload, f, indent=2)

                    self.logger.info(f"  Saved {filename}: {len(payload)} QA pairs")
                    downloaded = True
                    break
                except Exception as e:
                    self.logger.warning(f"    Failed: {e}")

            if not downloaded:
                if split in optional_splits:
                    self.logger.info(
                        f"  {filename} not available (test split is often not public), skipping."
                    )
                else:
                    self.logger.error(f"  Could not download {filename} from any URL")
                    have_required = False

        # Log stats for whatever we have
        self._log_scanqa_stats()

        if have_required:
            self.progress["completed_steps"].append("scanqa")
            self._save_progress()

        return have_required

    def _find_local_scanqa_annotations(self):
        """Search for ScanQA JSONs in subdirectories and copy them up if needed."""
        for split in SCANQA_SPLITS:
            filename = f"ScanQA_v1.0_{split}.json"
            dest = self.scanqa_dir / filename
            if dest.exists():
                continue

            # Search common subdirectory patterns
            candidates = [
                p for p in self.scanqa_dir.rglob(filename)
                if p.resolve() != dest.resolve()
            ]
            if candidates:
                src = candidates[0]
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(str(src.resolve()), str(dest))
                    self.logger.info(
                        f"  Found local {filename} at {src.relative_to(self.scanqa_dir)}"
                    )
                except Exception as e:
                    self.logger.warning(f"  Could not copy {src} -> {dest}: {e}")

    def _log_scanqa_stats(self):
        """Log ScanQA statistics."""
        for split in SCANQA_SPLITS:
            fpath = self.scanqa_dir / f"ScanQA_v1.0_{split}.json"
            if fpath.exists():
                try:
                    with open(fpath) as f:
                        data = json.load(f)
                    scenes = {item.get("scene_id", item.get("scan_id", "")) for item in data}
                    scenes.discard("")
                    self.logger.info(
                        f"  ScanQA {split}: {len(data)} QA pairs, {len(scenes)} unique scenes"
                    )
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Step 3: Preprocess ScanNet (optional)
    # ------------------------------------------------------------------
    def preprocess_scannet(self) -> bool:
        self.logger.info("=" * 60)
        self.logger.info("STEP 3: Preprocess ScanNet point clouds + images")
        self.logger.info("=" * 60)

        if self.scannet_root is None:
            self.logger.info("  --scannet-root not provided, skipping ScanNet preprocessing.")
            self.logger.info(
                "  Note: ScanNet requires a license agreement. Download it separately,"
            )
            self.logger.info("  then re-run with --scannet-root /path/to/scannet")
            return True

        if not self.scannet_root.exists():
            self.logger.error(f"  ScanNet root not found: {self.scannet_root}")
            return False

        self.pc_dir.mkdir(parents=True, exist_ok=True)
        self.img_dir.mkdir(parents=True, exist_ok=True)

        # Collect scene IDs from annotations
        scene_ids = self._collect_scene_ids()
        if not scene_ids:
            self.logger.error("  No scene IDs found in annotations.")
            return False

        self.logger.info(f"  Processing {len(scene_ids)} scenes from annotations")

        # Point clouds
        self._process_point_clouds(scene_ids)

        # Images
        self._process_images(scene_ids)

        self.progress["completed_steps"].append("scannet")
        self._save_progress()
        return True

    def _collect_scene_ids(self) -> List[str]:
        """Collect unique scene IDs from all downloaded annotation files."""
        scene_ids: Set[str] = set()

        # From SQA3D annotations
        for split in ["train", "val", "test"]:
            afile = self.sqa3d_dir / f"v1_balanced_sqa_annotations_{split}_scannetv2.json"
            if afile.exists():
                try:
                    with open(afile) as f:
                        data = json.load(f)
                    annots = data.get("annotations", data) if isinstance(data, dict) else data
                    if isinstance(annots, list):
                        for a in annots:
                            sid = a.get("scene_id", "")
                            if sid:
                                scene_ids.add(sid)
                except Exception:
                    pass

            # Also try question files for scene_id
            qfile = self.sqa3d_dir / f"v1_balanced_questions_{split}_scannetv2.json"
            if qfile.exists():
                try:
                    with open(qfile) as f:
                        data = json.load(f)
                    questions = data.get("questions", data) if isinstance(data, dict) else data
                    if isinstance(questions, list):
                        for q in questions:
                            sid = q.get("scene_id", "")
                            if sid:
                                scene_ids.add(sid)
                except Exception:
                    pass

        # From ScanQA annotations
        for split in SCANQA_SPLITS:
            fpath = self.scanqa_dir / f"ScanQA_v1.0_{split}.json"
            if fpath.exists():
                try:
                    with open(fpath) as f:
                        data = json.load(f)
                    for item in data:
                        sid = item.get("scene_id", item.get("scan_id", ""))
                        if sid:
                            scene_ids.add(sid)
                except Exception:
                    pass

        return sorted(scene_ids)

    def _process_point_clouds(self, scene_ids: List[str]):
        """Convert PLY meshes to NPY point clouds."""
        self.logger.info("  Processing point clouds (PLY -> NPY)...")

        # Check for PLY loading library
        use_trimesh = False
        try:
            import trimesh
            use_trimesh = True
            self.logger.info("    Using trimesh for PLY loading")
        except ImportError:
            try:
                from plyfile import PlyData
                self.logger.info("    Using plyfile for PLY loading")
            except ImportError:
                self.logger.error(
                    "    Need either 'trimesh' or 'plyfile'. "
                    "Install: pip install trimesh  OR  pip install plyfile"
                )
                return

        # Find PLY files in scannet_root
        # Common structures: scans/{scene_id}/{scene_id}_vh_clean_2.ply
        done = 0
        skipped = 0
        missing = 0
        failed = 0

        already_done = set(self.progress.get("scannet_scenes_done", []))

        for scene_id in tqdm(scene_ids, desc="Point clouds"):
            out_file = self.pc_dir / f"{scene_id}.npy"
            if out_file.exists():
                skipped += 1
                continue

            # Search for the PLY file
            ply_file = self._find_ply(scene_id)
            if ply_file is None:
                missing += 1
                continue

            try:
                if use_trimesh:
                    import trimesh
                    mesh = trimesh.load(str(ply_file), process=False)
                    vertices = np.array(mesh.vertices, dtype=np.float32)
                    if (
                        hasattr(mesh.visual, "vertex_colors")
                        and mesh.visual.vertex_colors is not None
                    ):
                        colors = (
                            np.array(mesh.visual.vertex_colors[:, :3], dtype=np.float32) / 255.0
                        )
                        points = np.concatenate([vertices, colors], axis=1)
                    else:
                        points = vertices
                else:
                    from plyfile import PlyData
                    plydata = PlyData.read(str(ply_file))
                    verts = plydata["vertex"]
                    x = np.array(verts["x"], dtype=np.float32)
                    y = np.array(verts["y"], dtype=np.float32)
                    z = np.array(verts["z"], dtype=np.float32)
                    try:
                        r = np.array(verts["red"], dtype=np.float32) / 255.0
                        g = np.array(verts["green"], dtype=np.float32) / 255.0
                        b = np.array(verts["blue"], dtype=np.float32) / 255.0
                        points = np.stack([x, y, z, r, g, b], axis=1)
                    except ValueError:
                        points = np.stack([x, y, z], axis=1)

                np.save(str(out_file), points)
                done += 1
                already_done.add(scene_id)
            except Exception as e:
                self.logger.warning(f"    Failed {scene_id}: {e}")
                failed += 1

        self.progress["scannet_scenes_done"] = list(already_done)
        self._save_progress()

        self.logger.info(
            f"  Point clouds: {done} created, {skipped} existed, "
            f"{missing} missing PLY, {failed} failed"
        )

    def _find_ply(self, scene_id: str) -> Optional[Path]:
        """Search for a scene's PLY file in scannet_root."""
        candidates = [
            self.scannet_root / "scans" / scene_id / f"{scene_id}_vh_clean_2.ply",
            self.scannet_root / scene_id / f"{scene_id}_vh_clean_2.ply",
            self.scannet_root / "scans" / scene_id / f"{scene_id}_vh_clean.ply",
            self.scannet_root / scene_id / f"{scene_id}_vh_clean.ply",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def _process_images(self, scene_ids: List[str]):
        """Pick a representative image per scene from frames_25k or sens files."""
        self.logger.info("  Picking representative images...")

        # Determine frame roots to search
        frame_roots: List[Path] = []
        if self.frames_dir is not None:
            frame_roots.append(self.frames_dir)
            frame_roots.append(self.frames_dir / "scannet_frames_25k")
            frame_roots.append(self.frames_dir / "frames_25k")

        # Also check common locations relative to scannet_root
        if self.scannet_root is not None:
            frame_roots.append(self.scannet_root / "frames_25k")
            frame_roots.append(self.scannet_root / "scannet_frames_25k")
            frame_roots.append(self.scannet_root / "tasks" / "scannet_frames_25k")

        if not frame_roots:
            self.logger.info("    No frames directory available, skipping image extraction.")
            return

        done = 0
        skipped = 0
        missing = 0

        for scene_id in tqdm(scene_ids, desc="Images"):
            out_file = self.img_dir / f"{scene_id}.jpg"
            if out_file.exists():
                skipped += 1
                continue

            found = False
            for froot in frame_roots:
                # Try color subdirectory first, then scene directory directly
                for subdir in [froot / scene_id / "color", froot / scene_id]:
                    if not subdir.exists():
                        continue
                    frames = sorted(subdir.glob("*.jpg")) + sorted(subdir.glob("*.png"))
                    if frames:
                        mid = len(frames) // 2
                        shutil.copy2(str(frames[mid]), str(out_file))
                        done += 1
                        found = True
                        break
                if found:
                    break

            if not found:
                missing += 1

        self.logger.info(f"  Images: {done} created, {skipped} existed, {missing} missing")

    # ------------------------------------------------------------------
    # Step 4: Validate scene coverage
    # ------------------------------------------------------------------
    def validate_coverage(self):
        self.logger.info("=" * 60)
        self.logger.info("STEP 4: Validate scene coverage")
        self.logger.info("=" * 60)

        # Get scene IDs per dataset
        sqa3d_scenes = self._get_sqa3d_scenes()
        scanqa_scenes = self._get_scanqa_scenes()
        all_scenes = sqa3d_scenes | scanqa_scenes

        # Check available point clouds and images
        available_pc = {f.stem for f in self.pc_dir.glob("*.npy")} if self.pc_dir.exists() else set()
        available_img = (
            {f.stem for f in self.img_dir.glob("*.jpg")} if self.img_dir.exists() else set()
        )

        self.logger.info(f"  Total unique scenes across all annotations: {len(all_scenes)}")
        self.logger.info(f"  Available point clouds: {len(available_pc)}")
        self.logger.info(f"  Available images: {len(available_img)}")

        if sqa3d_scenes:
            sqa3d_pc = sqa3d_scenes & available_pc
            sqa3d_img = sqa3d_scenes & available_img
            pct_pc = len(sqa3d_pc) / len(sqa3d_scenes) * 100 if sqa3d_scenes else 0
            pct_img = len(sqa3d_img) / len(sqa3d_scenes) * 100 if sqa3d_scenes else 0
            self.logger.info(
                f"  SQA3D ({len(sqa3d_scenes)} scenes): "
                f"point clouds {len(sqa3d_pc)}/{len(sqa3d_scenes)} ({pct_pc:.1f}%), "
                f"images {len(sqa3d_img)}/{len(sqa3d_scenes)} ({pct_img:.1f}%)"
            )

        if scanqa_scenes:
            scanqa_pc = scanqa_scenes & available_pc
            scanqa_img = scanqa_scenes & available_img
            pct_pc = len(scanqa_pc) / len(scanqa_scenes) * 100 if scanqa_scenes else 0
            pct_img = len(scanqa_img) / len(scanqa_scenes) * 100 if scanqa_scenes else 0
            self.logger.info(
                f"  ScanQA ({len(scanqa_scenes)} scenes): "
                f"point clouds {len(scanqa_pc)}/{len(scanqa_scenes)} ({pct_pc:.1f}%), "
                f"images {len(scanqa_img)}/{len(scanqa_scenes)} ({pct_img:.1f}%)"
            )

    def _get_sqa3d_scenes(self) -> Set[str]:
        scenes: Set[str] = set()
        for split in ["train", "val", "test"]:
            for pattern in [
                f"v1_balanced_sqa_annotations_{split}_scannetv2.json",
                f"v1_balanced_questions_{split}_scannetv2.json",
            ]:
                fpath = self.sqa3d_dir / pattern
                if fpath.exists():
                    try:
                        with open(fpath) as f:
                            data = json.load(f)
                        items = data.get("annotations", data) if isinstance(data, dict) else data
                        if not isinstance(items, list):
                            items = data.get("questions", []) if isinstance(data, dict) else []
                        for item in items:
                            sid = item.get("scene_id", "")
                            if sid:
                                scenes.add(sid)
                    except Exception:
                        pass
        return scenes

    def _get_scanqa_scenes(self) -> Set[str]:
        scenes: Set[str] = set()
        for split in SCANQA_SPLITS:
            fpath = self.scanqa_dir / f"ScanQA_v1.0_{split}.json"
            if fpath.exists():
                try:
                    with open(fpath) as f:
                        data = json.load(f)
                    for item in data:
                        sid = item.get("scene_id", item.get("scan_id", ""))
                        if sid:
                            scenes.add(sid)
                except Exception:
                    pass
        return scenes

    # ------------------------------------------------------------------
    # Step 5: Report statistics
    # ------------------------------------------------------------------
    def report_statistics(self):
        self.logger.info("=" * 60)
        self.logger.info("STEP 5: Final statistics")
        self.logger.info("=" * 60)

        # SQA3D stats
        self.logger.info("--- SQA3D ---")
        total_q = 0
        for split in ["train", "val", "test"]:
            qfile = self.sqa3d_dir / f"v1_balanced_questions_{split}_scannetv2.json"
            if qfile.exists():
                try:
                    with open(qfile) as f:
                        data = json.load(f)
                    questions = data.get("questions", data) if isinstance(data, dict) else data
                    count = len(questions) if isinstance(questions, (list, dict)) else 0
                    total_q += count
                    self.logger.info(f"  {split} questions: {count}")
                except Exception:
                    pass

        ad_file = self.sqa3d_dir / "answer_dict.json"
        if ad_file.exists():
            try:
                with open(ad_file) as f:
                    ad = json.load(f)
                self.logger.info(f"  Answer vocab size: {len(ad)}")
            except Exception:
                pass

        # Question type distribution from question files
        type_counts: Dict[str, int] = {}
        for split in ["train", "val", "test"]:
            qfile = self.sqa3d_dir / f"v1_balanced_questions_{split}_scannetv2.json"
            if qfile.exists():
                try:
                    with open(qfile) as f:
                        data = json.load(f)
                    questions = data.get("questions", data) if isinstance(data, dict) else data
                    if isinstance(questions, list):
                        for q in questions:
                            qt = q.get("question_type", q.get("type", "unknown"))
                            type_counts[qt] = type_counts.get(qt, 0) + 1
                except Exception:
                    pass
        if type_counts:
            self.logger.info(f"  Question types: {dict(sorted(type_counts.items()))}")

        # ScanQA stats
        self.logger.info("--- ScanQA ---")
        for split in SCANQA_SPLITS:
            fpath = self.scanqa_dir / f"ScanQA_v1.0_{split}.json"
            if fpath.exists():
                try:
                    with open(fpath) as f:
                        data = json.load(f)
                    scenes = {item.get("scene_id", item.get("scan_id", "")) for item in data}
                    scenes.discard("")
                    self.logger.info(
                        f"  {split}: {len(data)} QA pairs, {len(scenes)} unique scenes"
                    )
                except Exception:
                    pass

        # ScanNet stats
        self.logger.info("--- ScanNet ---")
        if self.pc_dir.exists():
            pc_files = list(self.pc_dir.glob("*.npy"))
            total_bytes = sum(f.stat().st_size for f in pc_files)
            self.logger.info(f"  Point clouds: {len(pc_files)} files ({total_bytes / 1e9:.2f} GB)")
        else:
            self.logger.info("  Point clouds: not available")

        if self.img_dir.exists():
            img_files = list(self.img_dir.glob("*.jpg"))
            total_bytes = sum(f.stat().st_size for f in img_files)
            self.logger.info(f"  Images: {len(img_files)} files ({total_bytes / 1e6:.1f} MB)")
        else:
            self.logger.info("  Images: not available")

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------
    def run(self) -> bool:
        self.logger.info("=" * 60)
        self.logger.info("3D QA Data Download Pipeline")
        self.logger.info("=" * 60)
        self.logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Data directory: {self.data_dir}")
        if self.scannet_root:
            self.logger.info(f"ScanNet root: {self.scannet_root}")
        if self.frames_dir:
            self.logger.info(f"Frames directory: {self.frames_dir}")
        self.logger.info("")

        if self.validate_only:
            self.validate_coverage()
            self.report_statistics()
            return True

        # Step 1: SQA3D
        if not self.download_sqa3d():
            self.logger.error("SQA3D download failed.")
            return False

        # Step 2: ScanQA (unless --sqa3d-only)
        if not self.sqa3d_only:
            if not self.download_scanqa():
                self.logger.warning("ScanQA download had issues (continuing).")

        # Step 3: ScanNet preprocessing (optional)
        if not self.sqa3d_only and not self.skip_scannet and self.scannet_root:
            if not self.preprocess_scannet():
                self.logger.warning("ScanNet preprocessing had issues (continuing).")

        # Step 4: Validate
        self.validate_coverage()

        # Step 5: Report
        self.report_statistics()

        self.logger.info("")
        self.logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("Done.")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare 3D QA datasets (SQA3D, ScanQA, ScanNet)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Root output directory (default: ./data)",
    )
    parser.add_argument(
        "--scannet-root",
        type=str,
        default=None,
        help="Path to raw ScanNet download (enables point cloud + image preprocessing)",
    )
    parser.add_argument(
        "--frames-dir",
        type=str,
        default=None,
        help="Path to scannet_frames_25k directory for image extraction",
    )
    parser.add_argument(
        "--sqa3d-only",
        action="store_true",
        help="Only download SQA3D annotations",
    )
    parser.add_argument(
        "--skip-scannet",
        action="store_true",
        help="Skip ScanNet preprocessing even if --scannet-root is provided",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run validation on existing data (no downloads)",
    )

    args = parser.parse_args()

    downloader = ThreeDQADownloader(
        data_dir=args.data_dir,
        scannet_root=args.scannet_root,
        frames_dir=args.frames_dir,
        sqa3d_only=args.sqa3d_only,
        skip_scannet=args.skip_scannet,
        validate_only=args.validate_only,
    )

    try:
        success = downloader.run()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        downloader.logger.info("\nInterrupted by user. Progress saved — run again to resume.")
        sys.exit(1)
    except Exception as e:
        downloader.logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
