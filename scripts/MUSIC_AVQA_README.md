# MUSIC-AVQA Dataset Download Script

Complete automated downloader for the MUSIC-AVQA dataset with YouTube throttling protection and cluster-friendly features.

## Features

✅ **Automated downloading** of MUSIC-AVQA metadata and videos
✅ **YouTube throttling protection** with rate limiting and exponential backoff
✅ **Resume capability** - interrupted downloads can be resumed
✅ **Parallel downloads** with configurable workers
✅ **Progress tracking** with detailed logging
✅ **Cluster-friendly** - can run in background with nohup
✅ **Retry logic** with exponential backoff for failed downloads
✅ **Audio extraction** - separate audio tracks in WAV format

## Prerequisites

### 1. Install yt-dlp

```bash
pip install yt-dlp
```

### 2. Install other dependencies

```bash
pip install requests tqdm
```

## Quick Start

### Interactive Mode (Local Machine)

```bash
cd scripts
python download_avqa.py --data-dir ./data/avqa --workers 4
```

### Background Mode (Cluster)

**Option 1: Using the shell script**
```bash
cd scripts
nohup ./run_avqa_download.sh ./data/avqa 4 50K 3.0 > avqa_download.out 2>&1 &
```

**Option 2: Direct Python call**
```bash
nohup python -u download_avqa.py \
    --data-dir ./data/avqa \
    --workers 4 \
    --rate-limit 50K \
    --sleep 3.0 \
    > avqa_download.out 2>&1 &
```

Monitor progress:
```bash
tail -f avqa_download.log
```

## Command Line Arguments

```
--data-dir PATH          Root directory for AVQA data (default: ./data/avqa)
--workers INT            Number of parallel workers (default: 4, max recommended: 8)
--rate-limit RATE        YouTube rate limit, e.g., '50K', '100K' (default: 50K)
--retry-attempts INT     Retry attempts for failed downloads (default: 5)
--sleep FLOAT            Base sleep time between downloads in seconds (default: 3.0)
--video-format STR       Video format (default: mp4)
--audio-format STR       Audio format (default: wav)
```

## Recommended Settings

### For Cluster/Server (Conservative)
```bash
python download_avqa.py \
    --workers 4 \
    --rate-limit 50K \
    --sleep 3.0 \
    --retry-attempts 5
```

This configuration:
- Uses 4 parallel workers
- Limits download speed to 50KB/s per video
- Waits 3-5 seconds between downloads (with random jitter)
- Will retry failed downloads up to 5 times with exponential backoff

### For Fast Connection (Aggressive)
```bash
python download_avqa.py \
    --workers 8 \
    --rate-limit 100K \
    --sleep 2.0 \
    --retry-attempts 3
```

⚠️ **Warning**: More aggressive settings may trigger YouTube rate limiting!

## Resume Interrupted Downloads

The script automatically tracks progress. If interrupted, simply run the same command again:

```bash
# First run (interrupted)
python download_avqa.py --data-dir ./data/avqa

# Resume from where it left off
python download_avqa.py --data-dir ./data/avqa
```

Progress is saved in: `./data/avqa/download_progress.json`

## Output Structure

```
data/avqa/
├── metadata/
│   ├── avqa-train.json          # Training split metadata
│   ├── avqa-val.json            # Validation split metadata
│   ├── avqa-test.json           # Test split metadata
│   └── video_ids.json           # Extracted video IDs
├── videos/
│   ├── VIDEO_ID_1.mp4
│   ├── VIDEO_ID_2.mp4
│   └── ...
├── audio/
│   ├── VIDEO_ID_1.wav
│   ├── VIDEO_ID_2.wav
│   └── ...
├── download_progress.json       # Progress tracking
├── failed_downloads.json        # Failed download attempts
└── avqa_download.log           # Detailed logs
```

## Monitoring Progress

### View download log
```bash
tail -f avqa_download.log
```

### Check progress
```bash
# Count downloaded videos
ls data/avqa/videos/*.mp4 | wc -l

# Check progress file
cat data/avqa/download_progress.json
```

### Check for failed downloads
```bash
cat data/avqa/failed_downloads.json
```

## Troubleshooting

### "yt-dlp not found"
```bash
pip install yt-dlp
# or
pip install --upgrade yt-dlp
```

### "Too many requests" or YouTube blocks
- Increase `--sleep` time (e.g., 5.0 or 10.0 seconds)
- Decrease `--workers` (e.g., 2 or 3)
- Lower `--rate-limit` (e.g., 25K)
- Wait a few hours and resume

### High failure rate
- Check internet connection
- Verify YouTube is accessible
- Try updating yt-dlp: `pip install --upgrade yt-dlp`
- Some videos may be unavailable/deleted - this is normal

### Resuming after failure
The script tracks failed downloads and will automatically retry them with exponential backoff. After all retries are exhausted, failed videos are listed in `failed_downloads.json`.

To manually retry failed downloads:
```bash
# Edit failed_downloads.json to remove video IDs you want to retry
# Then run the script again
python download_avqa.py --data-dir ./data/avqa
```

## Estimated Download Time

Assuming:
- ~1,000 unique videos in MUSIC-AVQA
- Average video size: 10-50 MB
- 4 workers with 50KB/s rate limit
- 3s sleep between downloads

**Estimated time**: 6-12 hours (conservative settings)

With more aggressive settings (8 workers, 100KB/s, 2s sleep): 3-6 hours

## Best Practices for Cluster Use

1. **Use conservative settings** to avoid getting blocked
2. **Run in background** with nohup
3. **Redirect output** to a file for monitoring
4. **Use screen or tmux** for long-running downloads
5. **Monitor logs** regularly to check progress
6. **Resume failed downloads** after fixing issues

Example with screen:
```bash
screen -S avqa_download
python download_avqa.py --data-dir ./data/avqa --workers 4
# Press Ctrl+A then D to detach
# Reattach with: screen -r avqa_download
```

## Verifying Downloads

After downloading, verify the dataset:

```bash
# Check counts
echo "Videos: $(ls data/avqa/videos/*.mp4 | wc -l)"
echo "Audio: $(ls data/avqa/audio/*.wav | wc -l)"
echo "Metadata: $(ls data/avqa/metadata/*.json | wc -l)"

# Check for corrupted files
find data/avqa/videos -type f -size 0
find data/avqa/audio -type f -size 0
```

## Citation

If you use the MUSIC-AVQA dataset, please cite:

```bibtex
@inproceedings{li2022learning,
  title={Learning to answer questions in dynamic audio-visual scenarios},
  author={Li, Guangyao and Wei, Yake and Tian, Yapeng and Xu, Chenliang and Wen, Ji-Rong and Hu, Di},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={19108--19118},
  year={2022}
}
```

## License

The MUSIC-AVQA dataset has its own license. Please refer to the [official repository](https://github.com/gewu-lab/MUSIC-AVQA) for licensing information.

## Support

For issues with:
- **This download script**: Open an issue in your SAFE repository
- **MUSIC-AVQA dataset**: Visit https://github.com/gewu-lab/MUSIC-AVQA
- **yt-dlp**: Visit https://github.com/yt-dlp/yt-dlp
