# LesionRec - Complete Demo Setup Guide

**Demo Date:** December 4, 2025
**Status:** Ready for testing

---

## 🎯 What You've Built

A multi-model AI system for acne analysis featuring:

1. **YOLOv8 Detection** ✅ (96.17% mAP@50 - EXCELLENT)
2. **YOLOv10 Detection** 🔄 (Currently training)
3. **Gemini Vision Analysis** ✅ (Natural language insights)
4. **Product Recommendations** ✅ (Personalized skincare routine)
5. **Streamlit Dashboard** ✅ (Professional web interface)

---

## 🚀 Quick Start

### Step 1: Activate Virtual Environment

```bash
source .venv/bin/activate  # Mac/Linux
# OR
.venv\Scripts\activate  # Windows
```

### Step 2: Set Up Gemini API Key

1. Get free API key: https://makersuite.google.com/app/apikey
2. Set environment variable:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

### Step 3: Launch Dashboard

```bash
streamlit run app/streamlit_app.py
```

The dashboard will open at: http://localhost:8501

---

## 📋 Features Overview

### 1. YOLOv8 Detection (COMPLETED ✅)

**Model:** `runs/detect/acne_yolov8_production/weights/best.pt`

**Performance:**
- mAP@50: **96.17%**
- mAP@50-95: **90.74%**
- Precision: **92.41%**
- Recall: **88.24%**

**What it does:**
- Detects 4 types of acne lesions
- Draws bounding boxes around each lesion
- Provides confidence scores

**Test it:**
```bash
python scripts/yolo_inference.py \
  --model runs/detect/acne_yolov8_production/weights/best.pt \
  --source path/to/image.jpg
```

---

### 2. YOLOv10 Detection (TRAINING 🔄)

**Model:** Training in progress

**Training Command:**
```bash
python scripts/train_yolov10.py \
  --model yolov10m.pt \
  --epochs 100 \
  --batch 16 \
  --name acne_yolov10_production \
  --device mps
```

**Check Progress:**
```bash
# View training logs
tail -f runs/detect/acne_yolov10_production/train.log

# Check results
cat runs/detect/acne_yolov10_production/results.csv
```

**When complete:**
- Model will be saved to `runs/detect/acne_yolov10_production/weights/best.pt`
- Dashboard will automatically detect and use it
- Compare with YOLOv8 performance

---

### 3. Gemini Vision Analysis (READY ✅)

**What it does:**
- Natural language analysis of acne severity
- Identifies lesion types
- Provides treatment recommendations
- Assesses skin type and concerns

**Test it:**
```bash
export GEMINI_API_KEY="your_key"

python scripts/gemini_analysis.py \
  --image path/to/image.jpg \
  --detailed
```

**Output:**
- Severity (mild/moderate/severe)
- Estimated lesion count
- Lesion types detected
- Skin concerns
- Treatment recommendations
- Summary

---

### 4. Product Recommendations (READY ✅)

**Database:** `data/products.json`

**What it does:**
- Maps acne type → specific products
- Provides cleanser, treatment, moisturizer recommendations
- Includes usage instructions, pricing, where to buy
- Customized routine based on severity

**Test it:**
```bash
python scripts/product_recommendations.py \
  --lesions '{"papules": 7, "pustules": 2, "comedones": 3}'
```

**Features:**
- Severity-based recommendations
- Budget options (budget/moderate/premium)
- Daily routine guidance
- Timeline for results
- When to see a doctor

---

### 5. Streamlit Dashboard (READY ✅)

**Launch:**
```bash
streamlit run app/streamlit_app.py
```

**Features:**
- Upload skin images
- Run multiple AI models simultaneously
- Side-by-side comparison
- Interactive results display
- Product recommendations
- Professional UI

**How to use:**
1. Upload image
2. Select models to run (YOLOv8, YOLOv10, Gemini)
3. Click "Analyze Skin"
4. View results in tabs
5. Get product recommendations

---

## 🎬 Demo Flow (Dec 4th)

### Introduction (2 mins)
- Problem: Acne affects 85% of people, assessment is subjective
- Solution: Multi-model AI system for objective analysis

### Live Demo (5-7 mins)

**1. Upload Test Image**
- Use sample acne image
- Show original vs annotated

**2. YOLOv8 Detection**
- Show bounding boxes
- Display lesion count & breakdown
- Highlight confidence scores

**3. YOLOv10 Detection** (if training complete)
- Compare with YOLOv8
- Show which model is more accurate
- Discuss speed vs accuracy trade-offs

**4. Gemini Vision Analysis**
- Show natural language insights
- Demonstrate severity assessment
- Highlight treatment recommendations

**5. Product Recommendations**
- Show personalized routine
- Explain why each product works
- Display timeline for results

### Technical Deep Dive (3 mins)
- Model architectures
- Training process & results
- Data augmentation strategies
- Performance metrics

### Q&A (2-3 mins)

---

## 📊 Key Talking Points

### Innovation
- **First multi-model approach**: Combines object detection + vision LLM
- **Personalized recommendations**: Not just detection, but actionable advice
- **State-of-the-art models**: YOLOv8 & YOLOv10 (latest YOLO versions)
- **Accessible**: Free Gemini API makes it scalable

### Technical Excellence
- **96.17% mAP@50**: Better than many research papers
- **Transfer learning**: Achieved great results with limited data (2,690 images)
- **Multi-scale detection**: Catches lesions of all sizes
- **Real-time inference**: Fast enough for production use

### Real-World Impact
- **Democratizes dermatology**: Anyone can get objective skin analysis
- **Early intervention**: Catches severe acne before permanent scarring
- **Cost-effective**: Saves doctor visits for mild/moderate cases
- **Educational**: Teaches users about acne types and treatments

### Future Work
- **Before/after generation**: AI-powered visualization of clear skin
- **Tracking over time**: Monitor treatment progress
- **Mobile app**: Deploy on smartphones
- **Prescription integration**: Connect with dermatologists for severe cases

---

## 🛠️ Troubleshooting

### Gemini API Not Working
```bash
# Check API key is set
echo $GEMINI_API_KEY

# Re-export if needed
export GEMINI_API_KEY="your_key_here"
```

### YOLOv10 Model Not Found
- Check if training is complete
- Look for: `runs/detect/acne_yolov10_production/weights/best.pt`
- If not ready, use YOLOv8 only for demo

### Streamlit Dashboard Errors
```bash
# Reinstall Streamlit
pip install --upgrade streamlit

# Check all dependencies
pip install -r requirements.txt
```

### Image Upload Issues
- Supported formats: JPG, JPEG, PNG
- Max file size: ~200MB (Streamlit default)
- Use clear, well-lit images

---

## 📁 Project Structure

```
LesionRec/
├── app/
│   └── streamlit_app.py          # Main dashboard
├── scripts/
│   ├── train_yolo.py             # YOLOv8 training
│   ├── train_yolov10.py          # YOLOv10 training
│   ├── yolo_inference.py         # YOLO testing
│   ├── gemini_analysis.py        # Gemini integration
│   └── product_recommendations.py # Product system
├── data/
│   ├── products.json             # Product database
│   └── yolo_dataset/             # Training data
├── runs/
│   └── detect/
│       ├── acne_yolov8_production/   # YOLOv8 results
│       └── acne_yolov10_production/  # YOLOv10 results
├── docs/
│   ├── FEATURE_IMPLEMENTATION_PLAN.md
│   └── TECHNICAL_DEEP_DIVE_PART2.md
└── README_DEMO.md                # This file
```

---

## 🎯 Success Checklist

Before demo day:

- [ ] YOLOv10 training complete
- [ ] Test dashboard with multiple images
- [ ] Prepare 3-5 sample images (varying severity)
- [ ] Practice demo walkthrough (under 10 mins)
- [ ] Test Gemini API (check rate limits)
- [ ] Backup plan if internet fails
- [ ] Screenshot key results for slides
- [ ] Prepare answers to common questions

---

## 💡 Demo Tips

1. **Start with a WOW moment**: Upload dramatic image, show all 3 models at once
2. **Tell a story**: "Imagine a teenager struggling with acne..."
3. **Show, don't tell**: Let the models speak for themselves
4. **Highlight unique features**: Multi-model comparison is YOUR innovation
5. **Be honest about limitations**: Mention when to see a real doctor
6. **End with impact**: How this helps real people

---

## 🤝 Credits

- **YOLOv8/v10**: Ultralytics (https://github.com/ultralytics)
- **Gemini Vision**: Google AI (https://ai.google.dev/)
- **Dataset**: Roboflow (acne detection dataset)
- **You**: Integration, product recommendations, dashboard

---

**Good luck with your demo! 🚀**

Questions? Issues? Check the troubleshooting section or review docs/
