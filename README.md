# LesionRec 🔬

**AI-Powered Multi-Model Acne Detection & Treatment Recommendation System**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive acne detection system combining YOLOv8, YOLOv10, and Google Gemini Vision AI to provide accurate lesion detection, severity assessment, and personalized skincare recommendations.

---

## 🎯 Overview

LesionRec addresses the challenge of objective acne assessment by leveraging multiple state-of-the-art AI models:

- **YOLOv8**: Fast, accurate object detection (96.17% mAP@50)
- **YOLOv10**: Latest YOLO with NMS-free architecture
- **Gemini Vision**: Natural language analysis and insights
- **Smart Recommendations**: Personalized product suggestions based on detected acne type

### Key Features

✨ **Multi-Model Detection** - Compare results from multiple AI models
🎯 **High Accuracy** - 96%+ mAP@50 on acne detection
💡 **Natural Language Insights** - Gemini Vision provides human-readable analysis
🛍️ **Product Recommendations** - Personalized skincare routine based on lesion type
📊 **Professional Dashboard** - Clean Streamlit interface for easy use
🔬 **Educational** - Includes learning modules explaining ML concepts

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Virtual environment (recommended)
- 8GB+ RAM
- Optional: GPU for faster inference (M1/M2 Mac supported via MPS)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/adonisja/LesionRec.git
   cd LesionRec
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Gemini API key** (optional but recommended)
   ```bash
   # Get free API key from: https://makersuite.google.com/app/apikey
   export GEMINI_API_KEY="your_api_key_here"
   ```

5. **Launch the dashboard**
   ```bash
   streamlit run app/streamlit_app.py
   # Or use the launcher script:
   ./launch_dashboard.sh
   ```

6. **Open your browser**
   - Navigate to: http://localhost:8501
   - Upload a skin image
   - Click "Analyze Skin"
   - View results and recommendations!

---

## 📋 Features in Detail

### 1. YOLOv8 Detection

**Performance:**
- mAP@50: **96.17%**
- mAP@50-95: **90.74%**
- Precision: **92.41%**
- Recall: **88.24%**

**Detects 4 Acne Types:**
- Comedones (blackheads/whiteheads)
- Papules (inflamed red bumps)
- Pustules (pus-filled lesions)
- Nodules (deep, painful cysts)

**Usage:**
```bash
python scripts/yolo_inference.py \
  --model runs/detect/acne_yolov8_production/weights/best.pt \
  --source path/to/image.jpg
```

### 2. YOLOv10 Detection

**Advantages over YOLOv8:**
- NMS-free architecture (faster, cleaner predictions)
- Better small object detection
- 20-30% faster inference

**Training:**
```bash
python scripts/train_yolov10.py \
  --model yolov10m.pt \
  --epochs 100 \
  --batch 16
```

### 3. Gemini Vision Analysis

**Provides:**
- Severity assessment (mild/moderate/severe)
- Natural language description
- Lesion type identification
- Skin type analysis
- Treatment recommendations

**Usage:**
```bash
export GEMINI_API_KEY="your_key"
python scripts/gemini_analysis.py --image path/to/image.jpg --detailed
```

### 4. Product Recommendations

**Smart matching based on:**
- Dominant lesion type
- Severity level
- Budget preference

**Includes:**
- Cleanser recommendations
- Treatment products (OTC & prescription)
- Moisturizers
- Daily routine guidance
- Timeline for results
- When to see a dermatologist

**Usage:**
```bash
python scripts/product_recommendations.py \
  --lesions '{"papules": 7, "pustules": 2, "comedones": 3}'
```

### 5. Streamlit Dashboard

**Interactive web interface featuring:**
- Multi-model comparison
- Side-by-side results
- Product recommendations
- Annotated detection images
- Confidence scores
- Model performance metrics

---

## 📊 Model Comparison

| Feature | YOLOv8 | YOLOv10 | Gemini Vision |
|---------|--------|---------|---------------|
| **Speed** | Fast | Faster | Moderate |
| **Accuracy** | 96.17% mAP@50 | TBD | N/A |
| **Output** | Bounding boxes | Bounding boxes | Natural language |
| **Use Case** | Precise detection | Real-time detection | Holistic assessment |
| **Cost** | Free (local) | Free (local) | Free tier available |

---

## 🗂️ Project Structure

```
LesionRec/
├── app/
│   └── streamlit_app.py          # Main dashboard application
├── scripts/
│   ├── train_yolo.py             # YOLOv8 training
│   ├── train_yolov10.py          # YOLOv10 training
│   ├── yolo_inference.py         # YOLO testing
│   ├── gemini_analysis.py        # Gemini Vision integration
│   ├── product_recommendations.py # Product recommendation engine
│   ├── prepare_yolo_dataset.py   # Dataset preparation
│   └── compare_models.py         # Model comparison utilities
├── data/
│   ├── products.json             # Skincare product database
│   ├── skin_features.csv         # Feature metadata
│   └── yolo_dataset/             # Training dataset (DVC tracked)
├── config/
│   ├── yolo_acne.yaml            # YOLOv8 config
│   └── yolov10_acne.yaml         # YOLOv10 config
├── learning_modules/
│   ├── 01_convolution_basics.py  # Interactive convolution demo
│   ├── 02_activation_functions.py # Activation functions explained
│   └── 03_neural_network_from_scratch.py # Build NN from scratch
├── docs/
│   ├── README_DEMO.md            # Demo setup guide
│   ├── TESTING_GUIDE.md          # Testing checklist
│   ├── FEATURE_IMPLEMENTATION_PLAN.md # Development roadmap
│   └── TECHNICAL_DEEP_DIVE_PART2.md # Advanced ML concepts
├── runs/
│   └── detect/
│       ├── acne_yolov8_production/   # YOLOv8 results
│       └── acne_yolov10_production/  # YOLOv10 results
├── requirements.txt              # Python dependencies
├── launch_dashboard.sh           # Quick launcher script
└── README.md                     # This file
```

---

## 🎓 Learning Resources

This project includes educational modules to understand the underlying ML concepts:

### Interactive Learning Modules

Run these to learn how neural networks work:

```bash
# 1. Convolution from scratch
python learning_modules/01_convolution_basics.py

# 2. Activation functions explained
python learning_modules/02_activation_functions.py

# 3. Neural network from scratch (NumPy only!)
python learning_modules/03_neural_network_from_scratch.py
```

### Documentation

- **[Technical Deep Dive](docs/TECHNICAL_DEEP_DIVE_PART2.md)** - Loss functions, metrics, transfer learning
- **[Complete Beginner's Guide](docs/COMPLETE_BEGINNERS_GUIDE.md)** - Step-by-step ML walkthrough
- **[Demo Setup](README_DEMO.md)** - Comprehensive demo preparation guide
- **[Testing Guide](TESTING_GUIDE.md)** - Dashboard testing checklist

---

## 🛠️ Training Your Own Model

### Prepare Dataset

```bash
# 1. Organize images and labels in YOLO format
python scripts/prepare_yolo_dataset.py

# 2. Verify dataset structure
data/yolo_dataset/
├── images/
│   ├── train/  (70% of data)
│   ├── val/    (15% of data)
│   └── test/   (15% of data)
└── labels/
    ├── train/
    ├── val/
    └── test/
```

### Train YOLOv8

```bash
python scripts/train_yolo.py \
  --model yolov8m.pt \
  --data config/yolo_acne.yaml \
  --epochs 100 \
  --batch 16 \
  --name my_acne_detector
```

### Train YOLOv10

```bash
python scripts/train_yolov10.py \
  --model yolov10m.pt \
  --data config/yolo_acne.yaml \
  --epochs 100 \
  --batch 16 \
  --name my_yolov10_detector
```

### Monitor Training

```bash
# View training results
tensorboard --logdir runs/detect/

# Check metrics
cat runs/detect/my_acne_detector/results.csv
```

---

## 📊 Performance Metrics

### YOLOv8 Results (Production Model)

| Metric | Score | Description |
|--------|-------|-------------|
| **mAP@50** | 96.17% | Mean Avg Precision at IoU=0.5 |
| **mAP@50-95** | 90.74% | mAP across IoU 0.5-0.95 |
| **Precision** | 92.41% | % of correct detections |
| **Recall** | 88.24% | % of ground truth detected |

### Per-Class Performance

| Class | Precision | Recall | mAP@50 |
|-------|-----------|--------|--------|
| Comedone | 93.2% | 89.5% | 95.8% |
| Papule | 94.5% | 90.1% | 96.2% |
| Pustule | 91.8% | 87.3% | 95.9% |
| Nodule | 90.2% | 86.1% | 96.8% |

---

## 🐛 Troubleshooting

### Common Issues

**1. PyTorch loading errors**
```bash
# Fix for PyTorch 2.6+ weights_only issue
# Already handled in app/streamlit_app.py
# If issues persist, the fix is already applied
```

**2. Gemini API errors**
```bash
# Verify API key is set
echo $GEMINI_API_KEY

# Check rate limits (free tier: 15 RPM)
# Wait a moment and retry
```

**3. Out of memory during training**
```bash
# Reduce batch size
python scripts/train_yolo.py --batch 8

# Or use smaller model
python scripts/train_yolo.py --model yolov8n.pt
```

**4. Dashboard won't start**
```bash
# Kill existing processes
pkill -f streamlit

# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Try manual launch
source .venv/bin/activate
streamlit run app/streamlit_app.py
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Areas for Contribution

- [ ] Before/after image generation (AI-powered acne removal)
- [ ] Mobile app integration
- [ ] Additional skin conditions (eczema, rosacea)
- [ ] Multi-language support
- [ ] User authentication and history tracking
- [ ] Integration with dermatologist platforms

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **Ultralytics** - YOLOv8/v10 framework ([ultralytics/ultralytics](https://github.com/ultralytics/ultralytics))
- **Google AI** - Gemini Vision API ([ai.google.dev](https://ai.google.dev/))
- **Roboflow** - Dataset management and augmentation
- **Streamlit** - Dashboard framework
- **PyTorch** - Deep learning framework

---

## 📧 Contact

**Akkeem** - [@adonisja](https://github.com/adonisja)

**Project Link:** [https://github.com/adonisja/LesionRec](https://github.com/adonisja/LesionRec)

---

## 🎯 Roadmap

### Current Features ✅
- [x] YOLOv8 detection with 96%+ accuracy
- [x] YOLOv10 integration
- [x] Gemini Vision analysis
- [x] Product recommendation system
- [x] Streamlit dashboard
- [x] Learning modules

### Upcoming Features 🚀
- [ ] Before/after image generation (Stable Diffusion inpainting)
- [ ] User authentication (Supabase)
- [ ] Analysis history tracking
- [ ] Mobile-responsive design
- [ ] Video analysis support
- [ ] Export to PDF reports

---

**Built with ❤️ for better skin health**

*Democratizing dermatology through AI*
