# General AVQA Dataset Downloader

Automated downloader for the **General AVQA** dataset (Yang et al., ACM MM 2022) - Audio-Visual Question Answering on everyday videos.

## Dataset Overview

- **Paper**: AVQA: A Dataset for Audio-Visual Question Answering on Videos (ACM MM 2022)
- **Videos**: 57,015 videos from VGG-Sound (daily audio-visual activities)
- **QA Pairs**: 57,335 question-answer pairs
- **Source**: https://mn.cs.tsinghua.edu.cn/avqa/
- **GitHub**: https://github.com/AlyssaYoung/AVQA

## Key Differences from MUSIC-AVQA

| Feature | General AVQA | MUSIC-AVQA |
|---------|--------------|------------|
| Videos | 57,015 (VGG-Sound) | 9,288 (music) |
| Domain | Daily activities | Musical performances |
| Classes | 309 audio classes | 22 instruments |
| Source | YouTube (diverse) | YouTube (music) |
| QA Pairs | 57,335 | 45,867 |

## Prerequisites

```bash
pip install yt-dlp requests tqdm
```

## Quick Start

### Download Everything (Cluster - Recommended)

```bash
# Conservative settings to avoid YouTube blocks
nohup python -u download_general_avqa.py \
    --data-dir ./data/avqa \
    --workers 4 \
    --rate-limit 50K \
    --sleep 3.0 \
    --quality 360 \
    > avqa_general_download.out 2>&1 &

# Monitor progress
tail -f avqa_general_download.log
```

### Interactive Mode (Local)

```bash
python download_general_avqa.py \
    --data-dir ./data/avqa \
    --workers 4
```

## Command Line Options

```
--data-dir PATH          Data directory (default: ./data/avqa)
--workers INT            Parallel workers (default: 4, max: 8)
--rate-limit RATE        YouTube rate limit (default: 50K)
--retry-attempts INT     Retry attempts (default: 5)
--sleep FLOAT            Sleep between downloads (default: 3.0s)
--quality STR            Video quality: 360, 480, 720 (default: 360)
```

## Download Pipeline

The script automatically performs these steps:

### 1. Download Metadata
Downloads from GitHub:
- `train_qa.json` - Training QA pairs
- `val_qa.json` - Validation QA pairs

### 2. Extract Video IDs
Parses metadata to extract required video IDs (VGG-Sound format)

### 3. Download VGG-Sound CSV
Downloads the VGG-Sound metadata CSV containing YouTube URLs

### 4. Map Video IDs to URLs
Matches required video IDs to YouTube URLs

### 5. Download Videos
Parallel download with:
- Rate limiting (50KB/s default)
- Throttling (3s delays)
- Retry logic
- Progress tracking

### 6. Extract Audio
Separate audio tracks in WAV format

## Output Structure

```
data/avqa/
├── metadata/
│   ├── train_qa.json                 # Training QA pairs (57,015 samples)
│   ├── val_qa.json                   # Validation QA pairs
│   ├── vggsound.csv                  # VGG-Sound metadata with URLs
│   ├── required_video_ids.json       # List of needed video IDs
│   └── video_url_mapping.json        # Video ID -> YouTube URL mapping
├── videos/
│   ├── ---1_cCGK4M_000001.mp4       # VGG-Sound format: {youtube_id}_{timestamp}
│   ├── ---4XfyD8M_000030.mp4
│   └── ...
├── audio/
│   ├── ---1_cCGK4M_000001.wav
│   ├── ---4XfyD8M_000030.wav
│   └── ...
├── features/                         # For extracted features (optional)
├── download_progress.json            # Progress tracking
├── failed_downloads.json             # Failed downloads
└── avqa_general_download.log         # Detailed logs
```

## Video ID Format

VGG-Sound videos use the format: `{youtube_id}_{start_timestamp}`

Example: `---1_cCGK4M_000001`
- YouTube ID: `---1_cCGK4M`
- Start time: `000001` seconds

## Estimated Download Time

For 57,015 videos with conservative settings:

- **Workers**: 4
- **Rate limit**: 50KB/s per video
- **Sleep**: 3-5s between downloads
- **Quality**: 360p (~5-20MB per video)

**Estimated time**:
- Best case: 48-72 hours
- Realistic: 3-5 days (accounting for failures, retries, throttling)
- Per video: ~5-10 seconds

**Disk space needed**: ~300-500GB for all videos + audio

## Monitoring Progress

### Check download stats
```bash
# Video count
ls data/avqa/videos/*.mp4 | wc -l

# Audio count
ls data/avqa/audio/*.wav | wc -l

# Check progress
cat data/avqa/download_progress.json | jq '.completed | length'

# Failed downloads
cat data/avqa/failed_downloads.json | jq 'length'
```

### Live monitoring
```bash
# Watch log file
tail -f avqa_general_download.log

# Real-time stats
watch -n 10 'echo "Videos: $(ls data/avqa/videos/*.mp4 2>/dev/null | wc -l)"; echo "Audio: $(ls data/avqa/audio/*.wav 2>/dev/null | wc -l)"'
```

## Resume After Interruption

The script automatically resumes:

```bash
# Just run the same command again
python download_general_avqa.py --data-dir ./data/avqa
```

Progress is saved in `download_progress.json` after each successful download.

## Handling Failures

### Some videos unavailable (normal)
Some YouTube videos may be deleted or unavailable. This is expected.

Typical availability: ~90-95% of videos

### High failure rate (>20%)
If many downloads fail:

1. **Increase delays**:
   ```bash
   --sleep 5.0  # or even 10.0
   ```

2. **Reduce workers**:
   ```bash
   --workers 2
   ```

3. **Lower rate limit**:
   ```bash
   --rate-limit 25K
   ```

4. **Update yt-dlp**:
   ```bash
   pip install --upgrade yt-dlp
   ```

### Retry failed downloads
Failed downloads are tracked. To retry:

```bash
# Edit failed_downloads.json to remove videos you want to retry
# Then run again
python download_general_avqa.py --data-dir ./data/avqa
```

## Cluster Best Practices

### Use tmux or screen
```bash
# Start session
tmux new -s avqa_download

# Run downloader
python download_general_avqa.py --data-dir ./data/avqa --workers 4

# Detach: Ctrl+B then D
# Reattach: tmux attach -t avqa_download
```

### Resource allocation
- **CPU**: 4-8 workers = 4-8 cores
- **Memory**: ~2-4GB should be sufficient
- **Network**: Stable connection with good bandwidth
- **Disk**: Ensure enough space (~500GB)

### Run in background
```bash
nohup python -u download_general_avqa.py \
    --data-dir /path/to/data/avqa \
    --workers 4 \
    --rate-limit 50K \
    > avqa.out 2>&1 &

# Save PID for later
echo $! > avqa_download.pid

# Kill if needed
kill $(cat avqa_download.pid)
```

## Troubleshooting

### VGG-Sound CSV download fails

If automatic download fails, manually download:

1. Visit: https://www.robots.ox.ac.uk/~vgg/data/vggsound/
2. Download `vggsound.csv`
3. Place in: `data/avqa/metadata/vggsound.csv`
4. Run script again

### "yt-dlp not found"
```bash
pip install yt-dlp
# or
pip install --upgrade yt-dlp
```

### Metadata download fails

Alternative: Clone the GitHub repo

```bash
git clone https://github.com/AlyssaYoung/AVQA.git
cp AVQA/data/train_qa.json data/avqa/metadata/
cp AVQA/data/val_qa.json data/avqa/metadata/
```

### YouTube throttling/blocking

Symptoms:
- Many "HTTP Error 429" messages
- "Too many requests" errors
- Very high failure rate

Solutions:
1. Increase `--sleep` to 5-10 seconds
2. Reduce `--workers` to 2-3
3. Lower `--rate-limit` to 25K
4. Wait a few hours before resuming
5. Consider using a VPN (if allowed by your institution)

## Verifying Downloads

### Check for corrupted files
```bash
# Find zero-byte files
find data/avqa/videos -type f -size 0
find data/avqa/audio -type f -size 0

# Remove corrupted files
find data/avqa/videos -type f -size 0 -delete
find data/avqa/audio -type f -size 0 -delete
```

### Verify JSON structure
```bash
# Check train_qa.json
python -m json.tool data/avqa/metadata/train_qa.json | head -20

# Count QA pairs
cat data/avqa/metadata/train_qa.json | jq 'length'
cat data/avqa/metadata/val_qa.json | jq 'length'
```

## Using the Dataset

After downloading, you can use the dataset with SAFE:

```python
from safe.data.datasets import AVQADataset

# Load dataset
dataset = AVQADataset(
    data_dir="./data/avqa",
    split="train",
    # ... other args
)
```

## Citation

If you use the General AVQA dataset, cite:

```bibtex
@inproceedings{yang2022avqa,
  title={AVQA: A Dataset for Audio-Visual Question Answering on Videos},
  author={Yang, Pinci and Wang, Xin and Duan, Xuguang and Chen, Hong and Hou, Runze and Jin, Chongyang and Xu, Mingqian},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={3480--3491},
  year={2022}
}
```

And VGG-Sound (source of videos):

```bibtex
@inproceedings{chen2020vggsound,
  title={VGGSound: A Large-scale Audio-Visual Dataset},
  author={Chen, Honglie and Xie, Weidi and Vedaldi, Andrea and Zisserman, Andrew},
  booktitle={ICASSP},
  year={2020}
}
```

## License

The General AVQA dataset and VGG-Sound have their own licenses. Please refer to:
- AVQA: https://github.com/AlyssaYoung/AVQA
- VGG-Sound: https://www.robots.ox.ac.uk/~vgg/data/vggsound/

## Support

For issues:
- **Download script**: Open issue in SAFE repo
- **AVQA dataset**: Visit https://github.com/AlyssaYoung/AVQA
- **VGG-Sound**: Visit https://www.robots.ox.ac.uk/~vgg/data/vggsound/
- **yt-dlp**: Visit https://github.com/yt-dlp/yt-dlp
