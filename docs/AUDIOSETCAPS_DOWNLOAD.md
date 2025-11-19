# AudioSetCaps Download Guide

Robust downloader for the 6M+ AudioSetCaps dataset with advanced anti-bot detection.

## Features

### Anti-Bot Detection Measures
- **Android Client API**: Uses YouTube's Android client API instead of web scraping (much harder to detect)
- **User Agent Rotation**: Rotates between Android, iOS, and desktop user agents
- **Human-like Timing**: Variable delays with random jitter (not constant intervals)
- **Adaptive Rate Limiting**: Automatically increases delays when bot detection is suspected
- **Smart Error Detection**: Detects bot-related errors (403, 429, captcha, etc.)
- **Cooldown Periods**: Adds 5-15s cooldown when bot detection is suspected
- **Break Simulation**: Takes 10-30s breaks every 100 downloads (simulates human fatigue)

### Robustness Features
- **Resume Capability**: SQLite database tracks all downloads, can resume from any interruption
- **Exponential Backoff**: 1s → 60s retry delays with jitter
- **Audio Verification**: Validates downloaded audio files
- **Parallel Workers**: Download multiple files simultaneously (default: 4 workers)
- **Progress Tracking**: Real-time stats and ETA
- **Detailed Logging**: All errors logged to `.download.log`

## Installation

```bash
pip install datasets yt-dlp soundfile
```

## Usage

### Test Run (100 samples)
```bash
python scripts/download_audiosetcaps_robust.py \
  --output-dir data/audiosetcaps \
  --max-samples 100 \
  --max-workers 2
```

### Full Download (6M+ samples)
```bash
python scripts/download_audiosetcaps_robust.py \
  --output-dir data/audiosetcaps \
  --max-workers 4 \
  --rate-limit 0.5
```

### Conservative Mode (Lower Bot Detection Risk)
```bash
python scripts/download_audiosetcaps_robust.py \
  --output-dir data/audiosetcaps \
  --max-workers 2 \
  --rate-limit 1.0 \
  --max-retries 3
```

### Resume Interrupted Download
```bash
# Automatically resumes from state database
python scripts/download_audiosetcaps_robust.py \
  --output-dir data/audiosetcaps \
  --skip-metadata
```

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir` | `data/audiosetcaps` | Output directory for audio files |
| `--max-workers` | `4` | Number of parallel download workers |
| `--max-retries` | `5` | Maximum retry attempts per file |
| `--rate-limit` | `0.5` | Delay between downloads (seconds) |
| `--max-samples` | `None` | Limit number of samples (for testing) |
| `--skip-metadata` | `False` | Skip metadata loading (resume mode) |

## Output Structure

```
data/audiosetcaps/
├── audio/
│   ├── -_7Xe9vD3Hpg_000004.wav
│   ├── -TL8Mp3xcUM_000000.wav
│   └── ...
├── .download_state.db          # SQLite tracking database
└── .download.log                # Detailed error logs
```

## State Database

The `.download_state.db` tracks:
- Completed downloads (skip on resume)
- Failed attempts with error messages
- Skipped files (exceeded max retries)
- Attempt counts and timestamps

**Check status:**
```bash
sqlite3 data/audiosetcaps/.download_state.db \
  "SELECT status, COUNT(*) FROM downloads GROUP BY status"
```

## Performance Expectations

### Conservative Settings (2 workers, 1s delay)
- **Rate**: ~7,200 files/hour
- **6M files**: ~35 days
- **Bot detection risk**: Very Low

### Balanced Settings (4 workers, 0.5s delay)
- **Rate**: ~14,400 files/hour
- **6M files**: ~17 days
- **Bot detection risk**: Low-Medium

### Aggressive Settings (8 workers, 0.3s delay)
- **Rate**: ~24,000 files/hour
- **6M files**: ~10 days
- **Bot detection risk**: Medium-High

## Handling Bot Detection

If you see many failures with messages like:
- "sign in to confirm"
- "too many requests"
- "unusual traffic"

The script will automatically:
1. Detect the bot block
2. Double the rate limit delay
3. Add 5-15s cooldown
4. Continue with slower rate

**Manual intervention:**
```bash
# Stop the script
# Wait 30-60 minutes
# Resume with more conservative settings
python scripts/download_audiosetcaps_robust.py \
  --output-dir data/audiosetcaps \
  --skip-metadata \
  --max-workers 2 \
  --rate-limit 2.0
```

## Monitoring Progress

The script logs progress every 100 samples:
```
Progress: 1500/6000000 (0.0%) | Rate: 4.2/s | ETA: 16 days, 12:34:56 | ✓ 1450 ✗ 20 ⊘ 30
```

- ✓ = Successfully downloaded
- ✗ = Failed (will retry)
- ⊘ = Skipped (exceeded max retries)

## Tips for Large-Scale Downloads

1. **Use screen/tmux**: Run in persistent session
   ```bash
   screen -S audiosetcaps
   python scripts/download_audiosetcaps_robust.py ...
   # Ctrl+A, D to detach
   ```

2. **Monitor disk space**: 6M files × 2MB = ~12TB

3. **Start conservative**: Begin with `--rate-limit 1.0`, let adaptive rate limiting optimize

4. **Multiple machines**: Run on different IPs with different subsets
   ```bash
   # Machine 1
   --max-samples 2000000

   # Machine 2
   # Modify script to skip first 2M samples
   ```

5. **Check logs regularly**: `tail -f data/audiosetcaps/.download.log`

## Troubleshooting

### High failure rate
- Increase `--rate-limit` to 1.0 or 2.0
- Reduce `--max-workers` to 2
- Check internet connection stability

### "Audio verification failed"
- Some videos may be corrupted or unavailable
- These are automatically skipped
- Check `.download.log` for details

### Database locked errors
- Only run one instance per output directory
- If crashed, may need to delete `.download_state.db-wal`

## Dataset Information

- **Source**: [JishengBai/AudioSetCaps](https://huggingface.co/datasets/JishengBai/AudioSetCaps)
- **Size**: ~6.1M audio clips (10 seconds each)
- **Total duration**: ~17,000 hours
- **Storage**: ~12TB (2MB per file average)
- **Captions**: High-quality generated captions using LLMs
