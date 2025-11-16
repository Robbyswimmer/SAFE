# SAFE Training/Eval Logging Guide

This document describes the comprehensive logging added to help diagnose issues with training and evaluation metrics.

## Logging Categories

### 1. Data Loading (`safe/data/datasets.py`)

#### Audio Loading
- **Location**: `_load_audio()` method
- **What's logged**:
  - ✓ First 3 successful audio loads with filename, sample rate, and waveform shape
  - ❌ First 5 failed audio loads with candidate paths tried
  - Summary message after initial logging phase

**Example output**:
```
[AudioLoad] ✓ Loaded: rqfQRErjfk8_170000.wav (sr=48000, shape=torch.Size([1, 240000]))
[AudioLoad] ✓ Loaded: rqu8iB22I_Y_5000.wav (sr=48000, shape=torch.Size([1, 240000]))
[AudioLoad] ✓ Audio loading working correctly (suppressing further success logs)
```

**Failed load example**:
```
[AudioLoad] ❌ File not found: rqfQRErjfk8_170000.wav
[AudioLoad]    Tried 3 paths:
[AudioLoad]      - /path/to/data/audiocaps/audio/val/rqfQRErjfk8_170000.wav
[AudioLoad]      - /path/to/data/audiocaps/audio/val/rqfQRErjfk8.wav
```

#### Batch Collation
- **Location**: `_collate_multimodal_batch()` function
- **What's logged**:
  - First 2 batches: size, audio/image counts, average references per sample
  - Summary message after initial logging

**Example output**:
```
[Collate] Batch 0: size=16, audio=16/16, images=0/16, avg_refs=5.0
[Collate] Batch 1: size=16, audio=16/16, images=0/16, avg_refs=5.0
[Collate] ✓ Collation working (suppressing further logs)
```

### 2. Evaluation Loop (`safe/training/stage_a.py`)

#### Batch Processing
- **Location**: Main evaluation loop (around line 1350)
- **What's logged**:
  - First 3 batches: label, size, audio flags, actual audio entries
  - First batch: sample question and reference count
  - Gate value for each of first 3 batches

**Example output**:
```
[EvalBatch 1] label=AUDIO_0, size=16, has_audio_flag=True, audio_entries_present=16
[EvalBatch 1] Sample question: 'What is happening in the audio?...'
[EvalBatch 1] Sample has 5 reference(s)
[EvalBatch 1] Gate set to: 1.0
```

### 3. Audio Encoding (`safe/models/safe_model.py`)

#### Audio Encoding Pipeline
- **Location**: `encode_audio()` method
- **What's logged**:
  - First 3 encoding calls: input type/shape, encoder type
  - Output shape, dtype, and device for first 3 calls

**Example output**:
```
[AudioEncode] Call 1: input=list[16], first_type=tuple, encoder=whisper
[AudioEncode] Output: tokens.shape=torch.Size([16, 128, 4096]), dtype=torch.bfloat16, device=cuda:0
[AudioEncode] Call 2: input=list[16], first_type=tuple, encoder=whisper
[AudioEncode] Output: tokens.shape=torch.Size([16, 128, 4096]), dtype=torch.bfloat16, device=cuda:0
```

### 4. Metrics Computation (`safe/training/stage_a.py`)

#### Audio Caption Samples
- **Location**: Evaluation loop (around line 1491)
- **What's logged**:
  - First 3 audio caption predictions vs references
  - Number of references for each sample
  - Accuracy score for each sample

**Example output**:
```
[AudioCaption Sample 1]
  Prediction: 'A large crowd cheers and applauds loudly'
  References (5): 'A large crowd cheers and applauds'
                  'An audience screams and gives applause'
  Accuracy: 0.845

[AudioCaption Sample 2]
  Prediction: 'People yell and laugh as an engine sputters'
  References (5): 'Popping and crackling repeats as men yell and laugh'
                  'A vehicle is running and crackling and popping as people laugh'
  Accuracy: 0.763
```

#### Metrics Summary
- **Location**: Before caption metrics computation (around line 1607)
- **What's logged**:
  - Total predictions and reference sets
  - Reference count statistics (avg, min, max)
  - Success confirmation

**Example output**:
```
[MetricsCompute] Computing caption metrics on 600 predictions with 600 reference sets
[MetricsCompute] Reference stats: avg=5.0, min=5, max=5
[MetricsCompute] ✓ Caption metrics computed successfully
```

## What to Look For

### Signs Audio is Loading Correctly ✓
1. `[AudioLoad] ✓ Loaded:` messages with real filenames
2. `[Collate]` shows `audio=X/X` where both numbers match batch size
3. `[EvalBatch]` shows `has_audio_flag=True` and `audio_entries_present > 0`
4. `[EvalBatch]` shows `Gate set to: 1.0` for audio batches
5. `[AudioEncode]` shows input as `list[N]` not empty

### Signs Audio is NOT Loading ❌
1. `[AudioLoad] ❌ File not found:` messages
2. `[Collate]` shows `audio=0/X`
3. `[EvalBatch]` shows `has_audio_flag=False` or `audio_entries_present=0`
4. `[EvalBatch]` shows `Gate set to: 0.0` for what should be audio batches
5. `[AudioEncode]` never appears or shows empty inputs

### Signs References are Correct ✓
1. `[Collate]` shows `avg_refs=5.0` (or ~5 for AudioCaps)
2. `[EvalBatch 1] Sample has 5 reference(s)`
3. `[AudioCaption Sample X]` shows `References (5):`
4. `[MetricsCompute]` shows `avg=5.0, min=5, max=5`

### Signs of Data/Metric Issues ❌
1. `avg_refs=1.0` - using wrong JSON file (single captions)
2. Predictions that are identical to references (data leakage)
3. Predictions that are just "No" or very short (VL mode instead of audio)
4. Very high accuracy (>0.95) on first few samples (suspiciously good)
5. `[AudioCaption Sample X]` predictions look like VQA answers not captions

## Testing on Cluster

Run a quick validation eval and check for these key indicators in the first ~50 lines of output:

```bash
# Should see all these patterns in the first minute of output:
grep -E "\[AudioLoad\]|\[Collate\]|\[EvalBatch\]|\[AudioEncode\]|\[AudioCaption\]|\[MetricsCompute\]" your_log_file.txt | head -50
```

Expected healthy output sequence:
1. `[AudioLoad] ✓ Loaded:` (3x)
2. `[Collate] Batch` (2-3x with audio counts)
3. `[EvalBatch 1]` with question sample
4. `[EvalBatch 1] Gate set to: 1.0`
5. `[AudioEncode] Call` (2-3x)
6. `[AudioCaption Sample]` (3x with predictions/references)
7. `[MetricsCompute]` with stats
