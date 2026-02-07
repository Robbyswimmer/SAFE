# AVQA Composition (Pre-FFN)

This experiment tests audio-visual modality composition for question answering
using Pre-FFN residual fusion. Primary ECCV composition benchmark.

## Dataset Summary

### MUSIC-AVQA (Primary)
- **Source**: Li et al., "Learning to Answer Questions in Dynamic Audio-Visual Scenarios" (CVPR 2022)
- **Videos**: 9,288 (7,422 Real + 1,866 Synthetic music performances)
- **QA Pairs**: 45,867
- **Domain**: Music performances, 22 instrument classes
- **Question Types**: Existential, counting, location, comparative, temporal
- **License**: See [MUSIC-AVQA GitHub](https://github.com/gewu-lab/MUSIC-AVQA)

### General AVQA (Secondary)
- **Source**: Yang et al., "AVQA: A Dataset for Audio-Visual Question Answering on Videos" (ACM MM 2022)
- **Videos**: 57,015 (daily activities from VGG-Sound)
- **QA Pairs**: 57,335
- **Domain**: Diverse daily activities, 309 audio classes

## Data Preparation Methodology

### Step 0: Obtain Raw Videos
MUSIC-AVQA videos are downloaded from YouTube or obtained from the dataset authors.
Videos are split into Real (7,422 clips from YouTube) and Synthetic (1,866 generated clips).

### Step 1: Extract Audio and Visual Frames
Audio and visual frames are extracted from raw video files using ffmpeg:

- **Audio**: Mono WAV at 16kHz (standard for speech/audio models)
  ```
  ffmpeg -i video.mp4 -ac 1 -ar 16000 -vn audio.wav
  ```
- **Visual frames**: Single keyframe (middle frame) extracted as JPEG
  ```
  ffmpeg -i video.mp4 -ss <midpoint> -vframes 1 frame.jpg
  ```

This is standard practice in audio-visual QA research. Single-frame extraction
follows SoTA methods (e.g., LAVISH, APE) that use a representative keyframe
rather than full video encoding, since most AV-QA questions reference the
overall scene rather than temporal dynamics.

### Step 2: Generate JSONL Manifests
Raw JSON metadata is converted to standardized JSONL format:
```json
{
  "sample_id": "music_avqa_train_0001",
  "question": "How many instruments are playing?",
  "answer": "2",
  "question_type": "counting",
  "audio_path": "audio/video_id.wav",
  "image_path": "frames/video_id.jpg"
}
```

Samples missing either audio or image are filtered with `--require-both`
to ensure all composition experiments have paired data.

### Step 3: Verify Data Integrity
After preparation, verify:
- Audio files: mono WAV, 16kHz, non-zero size
- Frame files: valid JPEG, non-zero size
- Manifest: all paths resolve, no missing media

## Cluster Data Layout

```
experiments/full_training/data/music_avqa/
  metadata/
    avqa-train.json       # Original MUSIC-AVQA train annotations
    avqa-val.json         # Original MUSIC-AVQA val annotations
    avqa-test.json        # Original MUSIC-AVQA test annotations
  videos/                 # Symlinks to Real + Synthetic videos (9,288)
  audio/                  # Extracted 16kHz mono WAV (9,288)
  frames/                 # Extracted middle-frame JPEG (9,288)
  manifests/
    train.jsonl           # Standardized train manifest
    validation.jsonl      # Standardized val manifest
```

## Reproduction Commands

### 1) Prepare data (extract audio/frames + build manifests)

```bash
sbatch scripts/prepare_music_avqa.sh
```

This runs ~2-4 hours on CPU. Extracts audio and frames, then builds JSONL manifests.

### 2) Train composition (all modalities)

```bash
DATA_ROOT=/data/SalmanAsif/RobbyMoseley/SAFE/SAFE/experiments/full_training/data/music_avqa \
  sbatch --gres=gpu:1 experiments/avqa_composition/scripts/train_preffn_music_avqa.sh
```

### 3) Modality ablation controls

```bash
# Audio-only
DATA_ROOT=/data/SalmanAsif/RobbyMoseley/SAFE/SAFE/experiments/full_training/data/music_avqa \
  sbatch --gres=gpu:1 \
  --export=ALL,TRAIN_MODALITY=audio,EVAL_MODALITIES=audio \
  experiments/avqa_composition/scripts/train_preffn_music_avqa.sh

# Image-only
DATA_ROOT=/data/SalmanAsif/RobbyMoseley/SAFE/SAFE/experiments/full_training/data/music_avqa \
  sbatch --gres=gpu:1 \
  --export=ALL,TRAIN_MODALITY=image,EVAL_MODALITIES=image \
  experiments/avqa_composition/scripts/train_preffn_music_avqa.sh
```

## Training Configuration

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Architecture | Pre-FFN residual | ECCV primary method |
| LLM | LLaVA-1.5-13B (frozen) | Preserves vision-language capability |
| Audio encoder | CLAP (frozen) | Best ESC-50 encoder |
| Fusion layers | 1,5,9,13,17,21 | 6-layer spread (ESC-50 optimal) |
| Audio tokens | 8 | ESC-50 optimal |
| Batch size | 2 | Memory constraint (13B model) |
| Epochs | 10 | AV-QA convergence typically < 10 epochs |
| Learning rate | 5e-5 | Standard for adapter fine-tuning |
| Precision | FP16 | Memory + speed |

## Experiment Design

### Conditions (3 x modality)
| Condition | Train | Eval | Purpose |
|-----------|-------|------|---------|
| Both (composition) | audio+image | audio+image | Main result |
| Audio-only | audio | audio | Ablation: audio sufficiency |
| Image-only | image | image | Ablation: image sufficiency |

### True Composition Architecture
```
LLaVA: processes input_ids + pixel_values (native image path)
SAFE: injects audio tokens as Pre-FFN residuals at fusion layers
Both modalities contribute in single forward pass
```

### Metrics
- **Exact Match (EM)**: Primary metric
- **Token F1**: Partial credit for multi-word answers
- **Per-question-type breakdown**: Existential, counting, location, comparative, temporal

### Success Criteria
- Composition (both) outperforms single-modality on questions requiring both audio and visual understanding
- Clear evidence of complementary information from each modality

## References

```bibtex
@inproceedings{li2022music,
  title={Learning to Answer Questions in Dynamic Audio-Visual Scenarios},
  author={Li, Guangyao and Wei, Yake and Tian, Yapeng and Xu, Chenliang and Wen, Ji-Rong and Hu, Di},
  booktitle={CVPR},
  year={2022}
}
```

