#!/usr/bin/env python3
"""Load NuScenes-QA dataset with error handling for corrupted files."""

from datasets import load_dataset

def load_nuscenes_qa(scene_type="day", split="train"):
    """Load dataset with streaming and error handling."""
    print(f"Loading NuScenes-QA {scene_type}/{split} with streaming...")

    samples = []
    ds = load_dataset(
        'KevinNotSmile/nuscenes-qa-mini',
        scene_type,
        split=split,
        streaming=True
    )

    try:
        for s in ds:
            samples.append(s)
    except Exception as e:
        print(f"Stopped after {len(samples)} samples (error: {type(e).__name__})")

    print(f"Loaded {len(samples)} samples")
    return samples


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-type", default="day", choices=["day", "night"])
    parser.add_argument("--split", default="train", choices=["train", "validation"])
    parser.add_argument("--save", type=str, help="Save to pickle file")
    args = parser.parse_args()

    samples = load_nuscenes_qa(args.scene_type, args.split)

    if samples:
        print(f"\nFirst sample:")
        print(f"  Question: {samples[0]['question']}")
        print(f"  Answer: {samples[0]['answer']}")

    if args.save:
        import pickle
        with open(args.save, 'wb') as f:
            pickle.dump(samples, f)
        print(f"\nSaved to {args.save}")
