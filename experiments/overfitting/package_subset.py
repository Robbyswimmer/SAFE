"""Package the overfitting subset defined in subset_manifest.json.

Usage (local machine with full AudioCaps data available):

    python -m experiments.overfitting.package_subset \
        --source-root data/audiocaps \
        --dest-root experiments/overfitting/data_pack

This copies only the audio files referenced in the manifest and stores a
self-contained package (manifest + captions). You can then tar `data_pack`
for transfer to the cluster.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class ManifestEntry:
    audiocap_id: str
    youtube_id: str
    start_time: float | None
    caption: str
    audio_relpath: str


@dataclass
class SubsetManifest:
    subset_name: str
    description: str
    seed: int
    source_dataset: str
    audio_rel_root: str
    metadata_columns: List[str]
    entries: List[ManifestEntry]

    @classmethod
    def load(cls, path: Path) -> "SubsetManifest":
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = [ManifestEntry(**entry) for entry in data["entries"]]
        return cls(
            subset_name=data["subset_name"],
            description=data["description"],
            seed=data["seed"],
            source_dataset=data["source_dataset"],
            audio_rel_root=data["audio_rel_root"],
            metadata_columns=data["metadata_columns"],
            entries=entries,
        )

    def to_dict(self) -> dict:
        return {
            "subset_name": self.subset_name,
            "description": self.description,
            "seed": self.seed,
            "source_dataset": self.source_dataset,
            "audio_rel_root": self.audio_rel_root,
            "metadata_columns": self.metadata_columns,
            "entries": [entry.__dict__ for entry in self.entries],
        }


def package_subset(
    manifest_path: Path,
    source_root: Path,
    dest_root: Path,
    val_manifest_path: Path | None = None,
) -> None:
    manifest = SubsetManifest.load(manifest_path)

    # Prepare destination directories
    audio_dest_root = dest_root / manifest.audio_rel_root
    audio_dest_root.mkdir(parents=True, exist_ok=True)

    captions = []
    copied_images = 0
    missing_files = []

    for entry in manifest.entries:
        rel_audio_path = Path(entry.audio_relpath)
        src_audio = source_root / rel_audio_path
        dst_audio = dest_root / rel_audio_path
        dst_audio.parent.mkdir(parents=True, exist_ok=True)

        if not src_audio.exists():
            missing_files.append(str(src_audio))
            continue

        shutil.copy2(src_audio, dst_audio)

        captions.append({
            "audiocap_id": entry.audiocap_id,
            "youtube_id": entry.youtube_id,
            "start_time": entry.start_time,
            "caption": entry.caption,
            "audio_relpath": str(rel_audio_path),
        })

    # Write manifest + captions to package
    (dest_root / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2), encoding="utf-8"
    )
    (dest_root / "captions.jsonl").write_text(
        "\n".join(json.dumps(cap) for cap in captions), encoding="utf-8"
    )

    # Copy original subset manifest for reference
    shutil.copy2(manifest_path, dest_root / "subset_manifest.json")

    if val_manifest_path and val_manifest_path.exists():
        val_manifest_data = json.loads(val_manifest_path.read_text(encoding="utf-8"))
        val_entries = val_manifest_data.get("entries", [])
        val_manifest_dest = dest_root / "val_manifest.json"
        val_manifest_dest.write_text(json.dumps(val_manifest_data, indent=2), encoding="utf-8")

        # Copy required images from COCO val2014
        coco_root = source_root.parent / "vqa" / "images"
        for entry in val_entries:
            image_filename = entry.get("image_filename")
            if not image_filename:
                continue
            candidate_paths = [
                coco_root / image_filename,
                coco_root / "val2014" / image_filename,
                coco_root / "train2014" / image_filename,
            ]
            src_image = next((p for p in candidate_paths if p.exists()), None)
            dst_image = dest_root / "images" / image_filename
            dst_image.parent.mkdir(parents=True, exist_ok=True)

            if src_image is not None and src_image.exists():
                shutil.copy2(src_image, dst_image)
                copied_images += 1
            else:
                missing_files.append(image_filename)

    readme_path = dest_root / "README.txt"
    readme_path.write_text(
        (
            f"Subset: {manifest.subset_name}\n"
            f"Source dataset: {manifest.source_dataset}\n"
            f"Total clips packaged: {len(captions)}\n"
            f"Validation images copied: {copied_images}\n"
        ),
        encoding="utf-8",
    )

    if missing_files:
        missing_log = dest_root / "missing_files.log"
        missing_log.write_text("\n".join(missing_files), encoding="utf-8")
        print(f"WARNING: {len(missing_files)} source files were missing. See {missing_log}")

    print(f"Packaged {len(captions)} clips to {dest_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package AudioCaps subset for SAFE overfitting experiments")
    parser.add_argument("--manifest", type=Path, default=Path("experiments/overfitting/subset_manifest.json"),
                        help="Manifest JSON describing the subset")
    parser.add_argument("--source-root", type=Path, default=Path("data/audiocaps"),
                        help="Root directory containing original AudioCaps data")
    parser.add_argument("--dest-root", type=Path, default=Path("experiments/overfitting/data_pack"),
                        help="Destination directory for packaged subset")
    parser.add_argument("--val-manifest", type=Path, default=Path("experiments/overfitting/val_manifest.json"),
                        help="Optional validation manifest to include in the package")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package_subset(args.manifest, args.source_root, args.dest_root, args.val_manifest)


if __name__ == "__main__":
    main()
