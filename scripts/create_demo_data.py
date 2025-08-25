#!/usr/bin/env python3
"""
Create minimal demo data for testing real dataset loading pipeline.
This creates synthetic media files to validate the data loading without requiring full datasets.
"""

import os
import numpy as np
import cv2
import librosa
import soundfile as sf
import json
from pathlib import Path

def create_demo_video(output_path, duration=10, fps=30, width=224, height=224):
    """Create a synthetic video file with moving patterns."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration * fps
    for i in range(total_frames):
        # Create synthetic frame with moving pattern
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Moving circle
        center_x = int(width/2 + 50 * np.sin(2 * np.pi * i / total_frames))
        center_y = int(height/2 + 30 * np.cos(2 * np.pi * i / total_frames))
        cv2.circle(frame, (center_x, center_y), 20, (255, 100, 50), -1)
        
        # Background gradient
        for y in range(height):
            for x in range(width):
                frame[y, x, 0] = min(255, int(128 + 127 * np.sin(x/width * np.pi)))
                frame[y, x, 1] = min(255, int(128 + 127 * np.cos(y/height * np.pi))) 
                frame[y, x, 2] = min(255, int(128 + 127 * np.sin((x+y)/(width+height) * np.pi)))
        
        out.write(frame)
    
    out.release()
    print(f"Created demo video: {output_path}")

def create_demo_audio(output_path, duration=10, sample_rate=16000):
    """Create synthetic audio with interesting patterns."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create synthetic audio with multiple components
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # A4 tone
    audio += 0.2 * np.sin(2 * np.pi * 880 * t)  # A5 tone
    audio += 0.1 * np.random.normal(0, 0.1, len(t))  # Noise
    
    # Add envelope
    envelope = np.exp(-t/duration * 2)
    audio = audio * envelope
    
    sf.write(output_path, audio, sample_rate)
    print(f"Created demo audio: {output_path}")

def create_avqa_metadata():
    """Create minimal MUSIC-AVQA style metadata."""
    metadata = []
    
    for i in range(5):  # Create 5 samples
        sample = {
            "video_id": f"sample_{i:03d}",
            "video_path": f"videos/sample_{i:03d}.mp4",
            "question": f"What instrument is playing in sample {i}?",
            "answer": ["piano", "guitar", "drums", "violin", "flute"][i],
            "question_type": "audio_dependent",
            "difficulty": "easy" if i < 3 else "medium"
        }
        metadata.append(sample)
    
    return metadata

def create_audiocaps_metadata():
    """Create minimal AudioCaps style metadata."""
    metadata = []
    
    captions = [
        "A piano melody plays softly",
        "Guitar strumming with rhythm",
        "Drums beating in steady tempo",
        "Violin playing classical music",
        "Flute producing gentle notes"
    ]
    
    for i, caption in enumerate(captions):
        sample = {
            "audio_id": f"audio_{i:03d}",
            "audio_path": f"audio/audio_{i:03d}.wav", 
            "caption": caption,
            "duration": 10.0
        }
        metadata.append(sample)
    
    return metadata

def create_vqa_metadata():
    """Create minimal VQA style metadata."""
    metadata = []
    
    for i in range(3):  # Create 3 validation samples
        sample = {
            "image_id": f"image_{i:03d}",
            "image_path": f"images/image_{i:03d}.jpg",
            "question": f"What color is dominant in image {i}?",
            "answer": ["red", "blue", "green"][i]
        }
        metadata.append(sample)
    
    return metadata

def main():
    """Create minimal demo dataset for testing."""
    print("Creating demo dataset for testing real data pipeline...")
    
    base_path = Path("./data")
    
    # Create directory structure
    (base_path / "videos").mkdir(parents=True, exist_ok=True)
    (base_path / "audio").mkdir(parents=True, exist_ok=True)  
    (base_path / "images").mkdir(parents=True, exist_ok=True)
    (base_path / "metadata").mkdir(parents=True, exist_ok=True)
    
    # Create synthetic media files
    print("\n1. Creating synthetic video files...")
    for i in range(5):
        video_path = base_path / "videos" / f"sample_{i:03d}.mp4"
        create_demo_video(str(video_path))
    
    print("\n2. Creating synthetic audio files...")
    for i in range(5):
        audio_path = base_path / "audio" / f"audio_{i:03d}.wav"
        create_demo_audio(str(audio_path))
    
    print("\n3. Creating synthetic images...")
    for i in range(3):
        # Create simple colored images
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]  # Red, Blue, Green
        image[:] = colors[i]
        
        # Add some pattern
        cv2.circle(image, (112, 112), 50, (255, 255, 255), 5)
        
        image_path = base_path / "images" / f"image_{i:03d}.jpg"
        cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"Created demo image: {image_path}")
    
    print("\n4. Creating metadata files...")
    
    # AVQA metadata
    avqa_data = create_avqa_metadata()
    with open(base_path / "metadata" / "avqa_train.json", 'w') as f:
        json.dump(avqa_data, f, indent=2)
    print(f"Created AVQA metadata: {len(avqa_data)} samples")
    
    # AudioCaps metadata
    audiocaps_data = create_audiocaps_metadata()
    with open(base_path / "metadata" / "audiocaps_train.json", 'w') as f:
        json.dump(audiocaps_data, f, indent=2)
    print(f"Created AudioCaps metadata: {len(audiocaps_data)} samples")
    
    # VQA metadata  
    vqa_data = create_vqa_metadata()
    with open(base_path / "metadata" / "vqa_val.json", 'w') as f:
        json.dump(vqa_data, f, indent=2)
    print(f"Created VQA metadata: {len(vqa_data)} samples")
    
    print(f"\n✓ Demo dataset created successfully!")
    print(f"📁 Location: {base_path.absolute()}")
    print(f"📊 Total samples: {len(avqa_data) + len(audiocaps_data)} train, {len(vqa_data)} val")
    print("\nYou can now test real data loading with:")
    print("python train_stage_a_curriculum.py --config demo --use-real-data --batch-size 2")

if __name__ == "__main__":
    main()