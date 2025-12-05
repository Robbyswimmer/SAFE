# SAFE Training - Quick Start

## 🚀 Run Training NOW

```bash
# Option 1: SLURM submission (recommended)
sbatch scripts/train_phase1.sh

# Option 2: Local execution
bash scripts/train_phase1.sh

# Option 3: Direct Python call
python train_safe.py \
    --model-config phase1 \
    --data-path ./data \
    --output-dir ./checkpoints/phase1 \
    --num-epochs 20 \
    --batch-size 4 \
    --gradient-accumulation-steps 32 \
    --fp16
```

## 📊 Expected Results (Phase 1)

| Metric | Target | Time |
|--------|--------|------|
| **CIDEr** | > 30 | After 20 epochs |
| **Training time** | 12-18h | Single GPU |
| **Memory** | ~22GB | With `--fp16` |

## 📁 Output Files

```
checkpoints/phase1/
├── args.json              # CLI arguments
├── history.json           # Training curves
├── checkpoint_last.pt     # Last checkpoint
└── checkpoint_best.pt     # Best checkpoint (highest CIDEr)
```

## 🔍 Monitor Progress

```bash
# Watch logs
tail -f logs/train_*.txt

# Check current metrics
cat checkpoints/phase1/history.json | python -m json.tool
```

## ✅ Success Criteria

- [ ] Training completes without OOM
- [ ] CIDEr score improves over epochs
- [ ] Final CIDEr > 30 (baseline was 18)
- [ ] Sample predictions are audio-relevant

## 📚 Full Documentation

- **Training guide**: `TRAINING_GUIDE.md`
- **Feature comparison**: `NEW_TRAINING_README.md`
- **Project status**: `SUMMARY.md`

## 🆘 Troubleshooting

**OOM error?**
```bash
# Reduce batch size
python train_safe.py ... --batch-size 2 --gradient-accumulation-steps 64
```

**Slow training?**
```bash
# Enable mixed precision
python train_safe.py ... --fp16
```

**CIDEr not improving?**
```bash
# Increase learning rate
python train_safe.py ... --learning-rate-projector 2e-3
```

---

**Ready?** → `sbatch scripts/train_phase1.sh`
