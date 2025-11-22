# Quick Start: YOLOv10 Acne Detection

Get up and running with YOLOv10 acne detection in **under 30 minutes**!

## 🚀 Ultra-Quick Start (3 Commands)

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install git+https://github.com/THU-MIG/yolov10.git

# 2. Prepare dataset (if you have raw images)
python scripts/prepare_yolo_dataset.py

# 3. Train your first model
python scripts/train_yolov10.py
```

That's it! Your model is training. ☕ Grab a coffee while it trains (~2-3 hours on GPU).

---

## 📋 Step-by-Step Guide

### Step 1: Installation (5 minutes)

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/LesionRec.git
cd LesionRec

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install YOLOv10
pip install git+https://github.com/THU-MIG/yolov10.git

# Verify installation
python -c "from ultralytics import YOLOv10; print('✓ YOLOv10 ready!')"
```

### Step 2: Get Data (10 minutes)

**Option A: Use existing DVC data**
```bash
# Add DVC to PATH (macOS)
export PATH="/Users/YOUR_USERNAME/Library/Python/3.9/bin:$PATH"

# Pull data from Google Drive
dvc pull
```

**Option B: Prepare your own dataset**
```bash
# Place your acne images in data/raw/acne/
# Then run:
python scripts/prepare_yolo_dataset.py
```

### Step 3: Train Model (2-3 hours)

**Quick training (nano model, fast)**:
```bash
python scripts/train_yolov10.py --model yolov10n.pt --epochs 50
```

**Production training (medium model, best balance)**:
```bash
python scripts/train_yolov10.py --model yolov10m.pt --epochs 150 --batch 16
```

**Monitor training**:
- Watch terminal output for loss curves
- Check `runs/detect/acne_yolov10/` for results
- Best model saved to `runs/detect/acne_yolov10/weights/best.pt`

### Step 4: Test Your Model (2 minutes)

**Test on a single image**:
```bash
python scripts/yolov10_inference.py \
  --model runs/detect/acne_yolov10/weights/best.pt \
  --source data/samples/acne_sample_1.jpg \
  --show
```

**Test on all samples**:
```bash
python scripts/yolov10_inference.py \
  --model runs/detect/acne_yolov10/weights/best.pt \
  --source data/samples/
```

**Save results as JSON**:
```bash
python scripts/yolov10_inference.py \
  --model runs/detect/acne_yolov10/weights/best.pt \
  --source data/samples/acne_sample_1.jpg \
  --save-json results.json
```

---

## 🎯 Common Use Cases

### 1. Quick Prototyping

**Goal**: Fast experiments with small model

```bash
# Train nano model (fastest, ~1 hour on GPU)
python scripts/train_yolov10.py \
  --model yolov10n.pt \
  --epochs 50 \
  --batch 32 \
  --cache

# Test immediately
python scripts/yolov10_inference.py \
  --model runs/detect/acne_yolov10/weights/best.pt \
  --source data/samples/
```

### 2. Production Deployment

**Goal**: Best accuracy for real-world use

```bash
# Train medium model (best balance)
python scripts/train_yolov10.py \
  --model yolov10m.pt \
  --epochs 150 \
  --batch 16 \
  --patience 25

# Test with higher confidence threshold
python scripts/yolov10_inference.py \
  --model runs/detect/acne_yolov10/weights/best.pt \
  --source data/samples/ \
  --conf 0.5 \
  --save-json production_results.json
```

### 3. Research & Clinical Use

**Goal**: Maximum accuracy, don't care about speed

```bash
# Train large model
python scripts/train_yolov10.py \
  --model yolov10l.pt \
  --epochs 200 \
  --batch 8 \
  --imgsz 640 \
  --patience 30

# Validate thoroughly
python scripts/yolov10_inference.py \
  --model runs/detect/acne_yolov10/weights/best.pt \
  --source data/yolo_dataset/images/test/ \
  --save-txt \
  --save-json clinical_results.json
```

### 4. Compare YOLOv8 vs YOLOv10

**Goal**: See which model is better for your use case

```bash
# Train both models
python scripts/train_yolov8.py --model yolov8m.pt --epochs 100 --name yolov8_acne
python scripts/train_yolov10.py --model yolov10m.pt --epochs 100 --name yolov10_acne

# Compare performance
python scripts/compare_models.py \
  --model1 runs/detect/yolov8_acne/weights/best.pt \
  --model1-name "YOLOv8m" \
  --model2 runs/detect/yolov10_acne/weights/best.pt \
  --model2-name "YOLOv10m" \
  --test-dir data/samples/ \
  --save-report comparison.json \
  --save-plot comparison.png
```

---

## 📊 Expected Results

### Training Metrics

After 100-150 epochs, you should see:

```
mAP@50: 0.75-0.85  (Good: >0.70, Excellent: >0.85)
mAP@50-95: 0.55-0.70
Precision: 0.80-0.90
Recall: 0.70-0.85
```

### Inference Speed

| Model | GPU (RTX 3060) | CPU |
|-------|---------------|-----|
| YOLOv10n | 120-140 FPS | 10-15 FPS |
| YOLOv10s | 80-100 FPS | 5-8 FPS |
| YOLOv10m | 50-70 FPS | 2-4 FPS |
| YOLOv10l | 30-50 FPS | 1-2 FPS |

### YOLOv10 vs YOLOv8

**YOLOv10 advantages**:
- ⚡ 20-30% faster inference
- 🎯 Better small object detection
- 🧹 Cleaner predictions (NMS-free)
- 👥 Better for clustered objects

**When to use YOLOv8**:
- More community support/tutorials
- Slightly more stable (older, more tested)

**Bottom line**: YOLOv10 is better for acne detection in most cases!

---

## 🐛 Troubleshooting

### "CUDA out of memory"
```bash
# Reduce batch size
python scripts/train_yolov10.py --batch 8

# Or reduce image size
python scripts/train_yolov10.py --imgsz 416

# Or train on CPU (slower)
python scripts/train_yolov10.py --device cpu
```

### "YOLOv10 not found"
```bash
# Reinstall YOLOv10
pip uninstall yolov10
pip install git+https://github.com/THU-MIG/yolov10.git

# If still fails, use YOLOv8 (works with same scripts)
python scripts/train_yolo.py  # Will use YOLOv8
```

### "No images found"
```bash
# Check if dataset exists
ls data/yolo_dataset/images/train/

# If empty, prepare dataset
python scripts/prepare_yolo_dataset.py

# Or check if DVC data is pulled
dvc pull
```

### Training is very slow
```bash
# Check if GPU is being used
python -c "import torch; print('GPU:', torch.cuda.is_available())"

# If False, install CUDA or use Google Colab
# If True but still slow, cache images:
python scripts/train_yolov10.py --cache
```

---

## 🎓 Next Steps

1. **Read the full guide**: [YOLOV10_BEGINNERS_GUIDE.md](docs/YOLOV10_BEGINNERS_GUIDE.md)
2. **Tune hyperparameters**: Edit [config/yolov10_acne.yaml](config/yolov10_acne.yaml)
3. **Collect more data**: More data = better model
4. **Try ensemble**: Combine YOLOv10 with other models
5. **Build an app**: Use FastAPI or Streamlit

---

## 📚 Documentation

- **Full Beginner's Guide**: [docs/YOLOV10_BEGINNERS_GUIDE.md](docs/YOLOV10_BEGINNERS_GUIDE.md)
- **Dataset Preparation**: [docs/LABELING_GUIDE.md](docs/LABELING_GUIDE.md)
- **Model Alternatives**: [docs/MODEL_ALTERNATIVES.md](docs/MODEL_ALTERNATIVES.md)
- **Original Setup**: [SETUP.md](SETUP.md)

---

## 🆘 Getting Help

**Questions?**
- Check the [FAQ](README.md#faq)
- Read [Troubleshooting Guide](docs/YOLOV10_BEGINNERS_GUIDE.md#troubleshooting)
- Open an issue on GitHub

**Contributing**:
- Fork the repo
- Make improvements
- Submit pull requests

---

## 📝 Quick Reference

### File Structure
```
LesionRec/
├── scripts/
│   ├── train_yolov10.py       # Train YOLOv10
│   ├── yolov10_inference.py   # Run inference
│   ├── compare_models.py      # Compare models
│   └── prepare_yolo_dataset.py # Prepare dataset
├── config/
│   ├── yolov10_acne.yaml      # YOLOv10 config
│   └── yolo_acne.yaml         # YOLOv8 config
├── data/
│   ├── raw/                   # Raw images
│   └── yolo_dataset/          # Prepared dataset
└── runs/
    └── detect/                # Training outputs
```

### Key Commands
```bash
# Install
pip install -r requirements.txt
pip install git+https://github.com/THU-MIG/yolov10.git

# Prepare data
python scripts/prepare_yolo_dataset.py

# Train (basic)
python scripts/train_yolov10.py

# Train (custom)
python scripts/train_yolov10.py --model yolov10m.pt --epochs 150

# Inference
python scripts/yolov10_inference.py --model best.pt --source image.jpg

# Compare models
python scripts/compare_models.py --model1 m1.pt --model2 m2.pt --test-dir data/samples/

# Resume training
python scripts/train_yolov10.py --resume runs/detect/acne_yolov10/weights/last.pt
```

---

**🎉 You're ready to start! Good luck!**
