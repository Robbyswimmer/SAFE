# Data Ethics and Fair Use Policy

**Project**: SAFE (Selectively Augmenting Frozen Encoders)
**Dataset**: AudioCaps (Google AudioSet subset)
**Last Updated**: 2025-10-05

---

## Fair Use Justification

This research project operates under **fair use** (17 U.S.C. § 107) for academic and non-commercial purposes:

### Fair Use Factors

1. **Purpose and Character**:
   - Academic research for audio-visual multimodal learning
   - Non-commercial educational use
   - Transformative use: raw audio → derived feature embeddings

2. **Nature of Copyrighted Work**:
   - YouTube videos are already publicly available
   - 10-second segments extracted (minimal portion)

3. **Amount Used**:
   - Only 10-second clips from much longer videos
   - Represents minimal necessary portion for research

4. **Effect on Market**:
   - No market substitution (embeddings cannot reconstruct audio)
   - No commercial distribution
   - Research findings benefit the academic community

### Key Ethical Practices

✅ **Store only derived features** (CLAP embeddings, not raw audio)
✅ **Delete raw audio** immediately after processing
✅ **Log all sources** with YouTube IDs and timestamps
✅ **Temporary downloads** only (never permanent storage)
✅ **Attribution** in all publications
✅ **No redistribution** of raw audio

---

## Data Pipeline

### 1. Metadata Collection

AudioCaps provides metadata CSVs with:
- YouTube video IDs
- Start timestamps
- Captions (annotations)

**Location**: `data/audiocaps/metadata/{train,val,test}.csv`

**Source**:
- Paper: [AudioCaps: Generating Captions for Audios in The Wild](https://aclanthology.org/N19-1011/) (Kim et al., NAACL 2019)
- Dataset: https://github.com/cdjkim/audiocaps

### 2. Audio Download (Temporary Only)

Raw audio segments are downloaded to **temporary directories** using `yt-dlp`:

```bash
# Downloads go to /tmp/ and are deleted after processing
python scripts/download_and_process_audiocaps.py --split train
```

**Process**:
1. Download 10s segment from YouTube
2. Extract CLAP embedding (512-dim vector)
3. Save embedding to HDF5
4. **Delete WAV file** before next download

**No permanent storage of raw audio.**

### 3. Feature Extraction

We extract **CLAP embeddings** using the frozen `laion/clap-htsat-unfused` model:

- **Input**: 10-second audio waveform (mono, 44.1kHz)
- **Output**: 512-dimensional embedding vector
- **Model**: Pre-trained CLAP (Contrastive Language-Audio Pretraining)

**Why embeddings, not raw audio?**
- Embeddings are **derived features** (transformative)
- Cannot reconstruct original audio
- 21,000× smaller (42 GB → 2 MB)
- Sufficient for multimodal alignment research

### 4. Storage Format

**HDF5 files** store only derived features:

```
data/audiocaps/features/
├── train_clap_embeddings.h5   # Embeddings only
├── train_sources.json          # Source attribution
├── val_clap_embeddings.h5
└── val_sources.json
```

**HDF5 structure**:
- `embeddings`: (N, 512) float32 array
- `youtube_ids`: (N,) string array
- Metadata: split, encoder model, num_samples

**Source logs** (`*_sources.json`):
```json
{
  "youtube_id": {
    "youtube_id": "abc123",
    "audiocap_id": "1234",
    "start_time": 30,
    "caption": "A dog barking",
    "checksum": "a1b2c3d4..."
  }
}
```

---

## Cleanup Procedures

### Existing Data Cleanup

For any existing WAV files from previous downloads:

```bash
# Dry run - test without deleting
python scripts/extract_audio_features.py --split train --dry-run

# Extract features and delete WAVs
python scripts/extract_audio_features.py --split train --delete-wavs
```

**Expected outcome**:
- ✅ CLAP embeddings extracted to HDF5
- ✅ Source metadata logged to JSON
- ✅ All WAV files deleted
- ✅ ~42 GB → ~2 MB compression

### Future Downloads

All future data collection uses auto-cleanup:

```bash
# Download, extract, auto-delete
python scripts/download_and_process_audiocaps.py --split train
```

**Process per file**:
1. Download to `/tmp/audiocaps_XXXXX/`
2. Extract embedding
3. Append to HDF5
4. Delete WAV immediately
5. Delete temp directory on completion

**No raw audio is ever retained.**

---

## Publication Guidelines

### Required Attribution

When publishing research using AudioCaps data:

**Cite AudioCaps paper**:
```bibtex
@inproceedings{kim2019audiocaps,
  title={AudioCaps: Generating Captions for Audios in The Wild},
  author={Kim, Chris Dongjoo and Kim, Byeongchang and Lee, Hyunmin and Kim, Gunhee},
  booktitle={NAACL-HLT},
  year={2019}
}
```

**Include fair use statement**:
> This research uses AudioCaps metadata to extract derived audio features
> (CLAP embeddings) under fair use for academic research (17 U.S.C. § 107).
> No raw audio files are stored or distributed. All source videos remain
> publicly available on YouTube, and proper attribution is provided.

**Data processing disclosure**:
> Audio data: We extract 10-second segments from YouTube videos specified in
> AudioCaps metadata, compute CLAP embeddings using the pre-trained
> laion/clap-htsat-unfused model, and delete all raw audio files. Only derived
> feature vectors (512-dimensional embeddings) are retained for training.

### What NOT to Include

❌ Do not redistribute raw audio files
❌ Do not share WAV files or downloadable audio
❌ Do not publish YouTube video URLs in bulk
❌ Do not claim ownership of source content

### What to Include

✅ Cite AudioCaps dataset and paper
✅ Document feature extraction process
✅ Explain fair use justification
✅ Share code for reproducibility (extraction scripts)
✅ Provide source logs (YouTube IDs + timestamps)

---

## Reproducibility

### For Other Researchers

To reproduce our feature extraction:

1. **Download metadata** from AudioCaps:
   ```bash
   # Place in data/audiocaps/metadata/
   wget https://github.com/cdjkim/audiocaps/raw/master/dataset/train.csv
   ```

2. **Install dependencies**:
   ```bash
   pip install yt-dlp h5py torchaudio transformers
   ```

3. **Extract features** (no raw audio retained):
   ```bash
   python scripts/download_and_process_audiocaps.py --split train
   ```

4. **Result**: HDF5 file with CLAP embeddings + source log

### Verification

Researchers can verify our feature extraction by:
- Checking SHA256 checksums in `*_sources.json`
- Re-extracting embeddings from same YouTube segments
- Comparing embedding vectors (should be identical for same CLAP model)

**Our source logs enable reproducibility without redistributing copyrighted audio.**

---

## Legal Compliance

### Copyright

- Source videos: Copyrighted by original YouTube creators
- AudioCaps metadata: Academic dataset (proper attribution provided)
- Our contribution: Feature extraction methodology, research findings

### Fair Use Compliance

We comply with fair use through:

1. **Minimal reproduction**: 10s clips only
2. **Transformation**: Audio → embeddings (non-reversible)
3. **Academic purpose**: Non-commercial research
4. **No market harm**: Cannot substitute original content
5. **Attribution**: Proper citation of all sources

### Data Retention

- Raw audio: **0 days** (deleted immediately)
- Derived features: Retained for research duration
- Source logs: Permanent (for attribution/reproducibility)

### Takedown Policy

If a YouTube video is removed or made private:
- Our embeddings remain (fair use transformation)
- Source log documents original source
- New researchers cannot extract from removed video
- No redistribution obligation (derived features only)

---

## Internal Guidelines

### For Lab Members

**When collecting new data**:
1. Always use `download_and_process_audiocaps.py` (auto-cleanup)
2. Never create permanent copies of WAV files
3. Verify temporary directories are cleaned up
4. Keep source logs updated
5. Document any changes to extraction process

**Storage policy**:
- HDF5 embeddings: ✅ Keep in `data/audiocaps/features/`
- Source JSON logs: ✅ Keep in `data/audiocaps/features/`
- WAV files: ❌ Delete immediately after extraction
- Temp directories: ❌ Clean up after each run

**Code review checklist**:
- [ ] No code writes WAV to non-temp locations
- [ ] All downloads use temp directories
- [ ] Cleanup happens even on errors (try/finally)
- [ ] Source metadata is logged
- [ ] Attribution is documented

---

## Audit Trail

All data processing is logged for transparency:

### Processing Logs

Location: `logs/data_processing/`

Contains:
- Download timestamps
- Extraction success/failure rates
- Source file mappings
- Checksum verifications

### Source Metadata

Location: `data/audiocaps/features/*_sources.json`

Contains:
- YouTube video ID
- Start timestamp
- AudioCaps ID
- Caption
- Embedding checksum

**This provides complete provenance without storing copyrighted content.**

---

## Summary

**Ethical Principles**:
1. Respect copyright through fair use
2. Store only derived, transformative features
3. Delete raw audio immediately
4. Provide full attribution
5. Enable reproducibility through documentation

**Technical Implementation**:
1. Temporary downloads only (`/tmp/`)
2. Immediate feature extraction (CLAP embeddings)
3. Auto-cleanup after processing
4. Compression: 42 GB → 2 MB
5. Complete source logging

**Legal Compliance**:
- Fair use for academic research ✅
- Non-commercial use ✅
- Transformative purpose ✅
- Proper attribution ✅
- No market substitution ✅

**This approach enables cutting-edge multimodal research while respecting intellectual property rights.**

---

## Contact

For questions about our data practices:
- Check this document first
- Review source code in `scripts/`
- Open issue on project repository

For takedown requests:
- We do not host or distribute raw audio
- We maintain only derived feature embeddings
- Contact project maintainers with specific concerns
