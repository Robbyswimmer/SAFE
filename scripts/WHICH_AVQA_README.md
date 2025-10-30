# Which AVQA Dataset Should You Use?

There are **two different AVQA datasets** available for audio-visual question answering research. Here's a guide to help you choose.

## Quick Comparison

| Feature | **General AVQA** ✅ | **MUSIC-AVQA** |
|---------|------------------|----------------|
| **Videos** | 57,015 | 9,288 |
| **QA Pairs** | 57,335 | 45,867 |
| **Domain** | Daily activities | Music performances |
| **Audio Classes** | 309 (diverse) | 22 (instruments) |
| **Source** | VGG-Sound | YouTube Music |
| **Year** | 2022 | 2022 |
| **Conference** | ACM MM | CVPR (Oral) |
| **Download Script** | `download_general_avqa.py` | `download_music_avqa.py` |

## General AVQA (Recommended for Most Use Cases)

**Best for**: General audio-visual reasoning, diverse scenarios

### Pros
✅ **Larger dataset**: 57,015 videos vs 9,288
✅ **More diverse**: 309 audio classes covering everyday sounds
✅ **Broader domain**: People talking, animals, vehicles, nature, etc.
✅ **Better for transfer learning**: Covers wider range of scenarios
✅ **More realistic**: Everyday audio-visual situations

### Cons
❌ Takes longer to download (~3-5 days for all videos)
❌ Requires more storage (~500GB)
❌ Some videos may be unavailable (YouTube deletions)

### Use General AVQA if:
- You want a comprehensive audio-visual QA benchmark
- You're building general-purpose models
- You need diversity in audio and visual content
- You want to test on everyday scenarios

### Download
```bash
python download_general_avqa.py --data-dir ./data/avqa --workers 4
```

See: `GENERAL_AVQA_README.md`

---

## MUSIC-AVQA

**Best for**: Music-specific research, instrument recognition

### Pros
✅ **Focused domain**: Specialized for music understanding
✅ **Faster download**: Smaller dataset (~6-12 hours)
✅ **Less storage**: ~100-150GB
✅ **High-quality annotations**: Carefully annotated music performances
✅ **Instrument focus**: Good for audio event localization

### Cons
❌ Smaller dataset: 9,288 videos
❌ Narrow domain: Only music performances
❌ Less diversity: 22 instrument classes only
❌ Limited transfer: May not generalize well to non-music domains

### Use MUSIC-AVQA if:
- Your research focuses on music understanding
- You need instrument-specific questions
- You have limited storage/time
- You're working on music video analysis

### Download
```bash
python download_music_avqa.py --data-dir ./data/music_avqa --workers 4
```

See: `MUSIC_AVQA_README.md`

---

## Detailed Comparison

### Dataset Size

**General AVQA**:
- Videos: 57,015
- QA pairs: 57,335
- Duration: Varies (10-30 seconds typically)
- Size: ~300-500GB

**MUSIC-AVQA**:
- Videos: 9,288
- QA pairs: 45,867 (more QA pairs per video)
- Duration: ~10 seconds
- Size: ~100-150GB

### Question Types

**General AVQA**:
- Existential: "Is there a dog barking?"
- Counting: "How many people are talking?"
- Location: "Where is the sound coming from?"
- Comparative: "Which sound is louder?"
- Temporal: "When does the car start?"
- Causal: "What causes the sound?"

**MUSIC-AVQA**:
- Existential: "Is there a piano?"
- Counting: "How many instruments?"
- Location: "Where is the violin?"
- Comparative: "Which instrument appears first?"
- Temporal: "When does the guitar start playing?"

### Audio Classes

**General AVQA** (309 classes):
```
Animals: dog, cat, bird, horse, lion...
Vehicles: car, train, airplane, motorcycle...
Nature: wind, rain, thunder, water...
Human: speech, laughing, crying, coughing...
Music: guitar, piano, drums, violin...
Household: door, alarm, phone, vacuum...
...and many more
```

**MUSIC-AVQA** (22 instruments):
```
accordion, acoustic_guitar, cello, clarinet,
erhu, flute, guzheng, piano, pipa, saxophone,
trumpet, tuba, ukulele, violin, xylophone, etc.
```

### Training Considerations

| Aspect | General AVQA | MUSIC-AVQA |
|--------|--------------|------------|
| **Training time** | Longer (more data) | Shorter |
| **GPU memory** | Higher (larger batches possible) | Lower |
| **Overfitting risk** | Lower (more data) | Higher |
| **Generalization** | Better (diverse) | Limited to music |
| **Convergence** | Slower | Faster |

---

## Download Time & Resources

### General AVQA
```
Workers: 4
Rate limit: 50KB/s
Sleep: 3-5s between downloads

Estimated time: 3-5 days
Disk space: 500GB
Network: ~500GB download
```

### MUSIC-AVQA
```
Workers: 4
Rate limit: 50KB/s
Sleep: 3-5s between downloads

Estimated time: 6-12 hours
Disk space: 150GB
Network: ~150GB download
```

---

## Recommendation by Use Case

### 🎯 General Audio-Visual Understanding
→ **Use General AVQA**

Good for:
- Multimodal foundation models
- Audio-visual grounding
- General scene understanding
- Video question answering

### 🎵 Music-Specific Applications
→ **Use MUSIC-AVQA**

Good for:
- Music video analysis
- Instrument recognition
- Music education tools
- Audio source separation

### 🔬 Research Comparisons
→ **Use Both**

For comprehensive evaluation:
- Train on General AVQA (diverse)
- Test on both datasets
- Report cross-dataset performance
- Analyze domain transfer

### 💻 Limited Resources
→ **Start with MUSIC-AVQA**

If you have:
- Limited storage (<200GB)
- Limited time (need quick experiments)
- Slow internet connection
- Smaller GPUs

Then download MUSIC-AVQA first, test your model, then scale to General AVQA if needed.

---

## Can I Use Both?

**Yes!** You can train on both datasets for maximum diversity:

```python
# Combine datasets for training
from safe.data.datasets import CombinedDataset

combined = CombinedDataset([
    AVQADataset(data_dir="./data/avqa", split="train"),
    MusicAVQADataset(data_dir="./data/music_avqa", split="train")
])
```

**Benefits**:
- More training data (66,303 videos)
- Better generalization
- Robust to domain shift
- Comprehensive evaluation

**Drawbacks**:
- Long download time (4-6 days)
- Large storage requirement (~650GB)
- Longer training time

---

## Quick Start Recommendations

### For SAFE Training (Your Use Case)

Since SAFE is designed for general multimodal understanding, **use General AVQA**:

```bash
cd scripts

# Download General AVQA
nohup python -u download_general_avqa.py \
    --data-dir /data/avqa \
    --workers 4 \
    --rate-limit 50K \
    > avqa_download.out 2>&1 &

# Monitor
tail -f avqa_general_download.log
```

### For Quick Experiments

Start with **MUSIC-AVQA** to validate your approach:

```bash
cd scripts

# Download MUSIC-AVQA first (faster)
nohup python -u download_music_avqa.py \
    --data-dir /data/music_avqa \
    --workers 4 \
    --rate-limit 50K \
    > music_avqa_download.out 2>&1 &
```

---

## Summary

| Your Goal | Recommended Dataset |
|-----------|-------------------|
| General research | **General AVQA** |
| Music research | **MUSIC-AVQA** |
| Limited resources | **MUSIC-AVQA** (start) |
| Production system | **General AVQA** |
| Quick prototyping | **MUSIC-AVQA** |
| Best performance | **Both combined** |

---

## Getting Help

- **General AVQA**: See `GENERAL_AVQA_README.md`
- **MUSIC-AVQA**: See `MUSIC_AVQA_README.md`
- **Issues**: Open an issue in the SAFE repository

## Citations

**General AVQA**:
```bibtex
@inproceedings{yang2022avqa,
  title={AVQA: A Dataset for Audio-Visual Question Answering on Videos},
  author={Yang, Pinci and Wang, Xin and Duan, Xuguang and others},
  booktitle={ACM MM},
  year={2022}
}
```

**MUSIC-AVQA**:
```bibtex
@inproceedings{li2022music,
  title={Learning to Answer Questions in Dynamic Audio-Visual Scenarios},
  author={Li, Guangyao and Wei, Yake and Tian, Yapeng and others},
  booktitle={CVPR},
  year={2022}
}
```
