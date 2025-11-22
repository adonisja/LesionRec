# 🎓 Complete Training Walkthrough: Acne Detection System

**Student**: Akkeem
**Date**: November 16, 2025
**Goal**: Train, test, and demo an acne detection AI system

---

## 📚 Table of Contents

1. [Phase 1: Environment Setup](#phase-1)
2. [Phase 2: Dataset Preparation](#phase-2)
3. [Phase 3: Training Models](#phase-3)
4. [Phase 4: Testing & Evaluation](#phase-4)
5. [Phase 5: Model Comparison](#phase-5)
6. [Phase 6: Building MVP Demo](#phase-6)
7. [Phase 7: Presentation Prep](#phase-7)

---

## PHASE 1: Environment Setup {#phase-1}

### ✅ Current Status
- Python 3.9.6 installed ✓
- pip3 available ✓
- Raw acne data available (2692 images) ✓
- Sample images available (10 images) ✓

### 🎯 What We're Doing
Creating a clean Python environment with all necessary AI libraries.

### 📖 Why This Matters
**Analogy**: Like setting up a science lab before an experiment. We need all the right equipment (libraries) in the right place.

**What happens without this**:
- Library conflicts (like mixing incompatible chemicals)
- Version mismatches (using old tools)
- System-wide pollution (installing everything globally)

### 🔧 Step 1.1: Create Virtual Environment

**Command**:
```bash
python3 -m venv venv
```

**What this does**: Creates an isolated Python environment in a folder called `venv`

**Think of it as**: Your own private laboratory where you can install whatever you need without affecting anything else on your computer.

**You'll see**: A new `venv/` folder appear in your project directory

### 🔧 Step 1.2: Activate Virtual Environment

**Command**:
```bash
source venv/bin/activate
```

**What this does**: "Enters" your private laboratory

**You'll see**: `(venv)` appear at the start of your terminal prompt

**Example**:
```
Before: akkeem@Macbook LesionRec %
After:  (venv) akkeem@Macbook LesionRec %
```

### 🔧 Step 1.3: Upgrade pip

**Command**:
```bash
pip install --upgrade pip
```

**What this does**: Updates the package installer to the latest version

**Why**: Newer pip versions are faster and handle dependencies better

**You'll see**: Download progress and "Successfully installed pip-XX.X.X"

### 🔧 Step 1.4: Install Core Dependencies

**Command**:
```bash
pip install torch torchvision opencv-python numpy pandas matplotlib
```

**What this does**: Installs the fundamental AI and image processing libraries

**Breaking it down**:
- `torch`: PyTorch - The AI framework (like the engine of a car)
- `torchvision`: Image-specific AI tools (steering wheel, cameras)
- `opencv-python`: Image manipulation (photo editing tools)
- `numpy`: Fast math operations (calculator)
- `pandas`: Data organization (spreadsheet)
- `matplotlib`: Creating graphs (charting tools)

**Time**: 2-5 minutes
**You'll see**: Lots of downloading and "Successfully installed..." messages

### 🔧 Step 1.5: Install YOLO Libraries

**Command**:
```bash
pip install ultralytics
```

**What this does**: Installs the YOLO object detection framework

**What is YOLO?**: "You Only Look Once" - an AI model that can find objects in images really fast

**Analogy**: Like having a super-fast detective who can spot all the acne in a photo in milliseconds

**Time**: 1-2 minutes

### 🔧 Step 1.6: Install YOLOv10 (Latest)

**Command**:
```bash
pip install git+https://github.com/THU-MIG/yolov10.git
```

**What this does**: Installs the newest YOLO version (v10) directly from its source code

**Why from GitHub?**: It's so new it's not in the standard package library yet

**Think of it as**: Getting a brand-new iPhone model before it's in stores

**Time**: 1-2 minutes

### 🔧 Step 1.7: Verify Installation

**Command**:
```bash
python -c "from ultralytics import YOLO; print('✓ YOLOv8 ready!')"
python -c "from ultralytics import YOLOv10; print('✓ YOLOv10 ready!')"
```

**What this does**: Quick test to make sure everything installed correctly

**You should see**:
```
✓ YOLOv8 ready!
✓ YOLOv10 ready!
```

**If you see errors**: Don't panic! We'll troubleshoot together.

### ✅ Checkpoint 1
At this point, you should have:
- [x] Virtual environment created and activated
- [x] All AI libraries installed
- [x] Both YOLO versions working

**Time elapsed**: ~10-15 minutes

---

## PHASE 2: Dataset Preparation {#phase-2}

### 🎯 What We're Doing
Converting raw acne images into a format that AI models can learn from.

### 📖 Understanding the Dataset

**What you have**:
- 2,692 raw acne images in `data/raw/acne/`
- 10 sample images in `data/samples/` for testing

**What we need to create**:
```
data/yolo_dataset/
├── images/
│   ├── train/      (70% of images - for learning)
│   ├── val/        (15% of images - for tuning)
│   └── test/       (15% of images - for final testing)
└── labels/
    ├── train/      (bounding boxes for training images)
    ├── val/        (bounding boxes for validation images)
    └── test/       (bounding boxes for test images)
```

**Analogy**: Like organizing a textbook:
- **Training set**: Practice problems you study from
- **Validation set**: Quiz questions to check understanding
- **Test set**: Final exam (never seen before)

### 🔧 Step 2.1: Check Current Dataset Status

**Command**:
```bash
ls -lh data/raw/acne/ | wc -l
```

**What this does**: Counts how many acne images we have

**You should see**: Around 2,692

### 🔧 Step 2.2: Auto-Generate Labels Using Pre-trained Model

**Why?**: Manually labeling 2,692 images would take weeks! We'll use an existing AI model to do it automatically.

**Command**:
```bash
python scripts/generate_labels.py \
  --api-key YOUR_ROBOFLOW_API_KEY \
  --input data/raw/acne \
  --output data/labels/acne_labels.csv
```

**⚠️ WAIT**: Do you have a Roboflow API key?

**If NO** (which is fine!), we have two options:

**Option A: Quick Demo Path** (Recommended for presentation)
- Use the 10 sample images that already have labels
- Faster to train (10 minutes vs 2 hours)
- Good enough to show the concept
- We'll do this first, then explain how to scale up

**Option B: Full Production Path**
- Get free Roboflow account
- Generate labels for all 2,692 images
- Train production-quality model
- Takes longer but more impressive results

**Which would you prefer?** For now, let's assume Option A (quick demo), then I'll show you Option B.

### 🔧 Step 2.3: Quick Demo Dataset (Option A)

Since we have sample images, let's create a tiny dataset for quick training:

**Command**:
```bash
python scripts/prepare_quick_demo_dataset.py
```

**⚠️ Note**: This script doesn't exist yet. Let me create it for you!

### 📝 What Happens Next

I'll create a special script that:
1. Uses your 10 sample images
2. Creates synthetic labels (simulated bounding boxes)
3. Splits them into train/val/test
4. Formats everything for YOLO

This lets you:
- Train a model in 10-15 minutes instead of 2 hours
- See the entire process quickly
- Understand how it works
- Then scale up to full dataset later

---

## PHASE 3: Training Models {#phase-3}

### 🎯 What We're Doing
Teaching AI models to recognize acne by showing them many examples.

### 📖 How AI Learning Works

**Analogy**: Teaching a child to recognize dogs:
1. **Show examples**: "This is a dog" (thousands of times)
2. **Child guesses**: "Is this a dog?"
3. **Correct mistakes**: "Yes! / No, that's a cat"
4. **Repeat**: Child gets better each time

**In AI terms**:
1. **Epoch**: One complete pass through all training images
2. **Loss**: How wrong the model is (lower = better)
3. **Batch**: How many images to show at once
4. **Learning rate**: How fast the model adjusts

### 🔧 Step 3.1: Train YOLOv8 (Baseline Model)

**Command**:
```bash
python scripts/train_yolo.py \
  --model yolov8n.pt \
  --data config/yolo_acne.yaml \
  --epochs 50 \
  --batch 8 \
  --name yolov8_demo
```

**Breaking down the flags**:
- `--model yolov8n.pt`: Use the "nano" version (smallest, fastest)
- `--data config/yolo_acne.yaml`: Where to find the dataset
- `--epochs 50`: Go through the data 50 times
- `--batch 8`: Show 8 images at once
- `--name yolov8_demo`: Call this experiment "yolov8_demo"

**What you'll see**:
```
Epoch  GPU_mem  box_loss  cls_loss  dfl_loss  Instances  Size
1/50   0.5G     1.234     0.856     1.432     12         640
2/50   0.5G     1.102     0.743     1.298     15         640
...
```

**Reading the output**:
- **Epoch**: Which round of learning (1 to 50)
- **box_loss**: How wrong the bounding boxes are (should decrease)
- **cls_loss**: How wrong the classifications are (should decrease)
- **Instances**: Number of acne lesions in these images

**Good signs**:
- ✅ Losses go down over time
- ✅ No error messages
- ✅ Process completes

**Bad signs**:
- ❌ Losses stay flat or increase
- ❌ "Out of memory" errors
- ❌ NaN (Not a Number) values

**Time**: 10-20 minutes for demo dataset, 2-3 hours for full dataset

### 🔧 Step 3.2: Train YOLOv10 (Latest Model)

**Command**:
```bash
python scripts/train_yolov10.py \
  --model yolov10n.pt \
  --data config/yolov10_acne.yaml \
  --epochs 50 \
  --batch 8 \
  --name yolov10_demo
```

**Why train two models?**
1. **Comparison**: See which works better
2. **Learning**: Understand the improvements in v10
3. **Presentation**: Show students the evolution of AI

**Differences you'll notice**:
- YOLOv10 might train slightly faster
- Different architecture (internal structure)
- Potentially better accuracy on small objects (acne lesions)

**Time**: Same as YOLOv8 (10-20 minutes)

### ✅ Checkpoint 2
After both models finish training, you'll have:
- `runs/detect/yolov8_demo/weights/best.pt` (YOLOv8 model)
- `runs/detect/yolov10_demo/weights/best.pt` (YOLOv10 model)

These `.pt` files are your trained AI models!

---

## PHASE 4: Testing & Evaluation {#phase-4}

### 🎯 What We're Doing
Seeing how well our trained models actually work on new images.

### 🔧 Step 4.1: Test YOLOv8

**Command**:
```bash
python scripts/yolo_inference.py \
  --model runs/detect/yolov8_demo/weights/best.pt \
  --source data/samples/acne_sample_1.jpg \
  --save-json yolov8_results.json
```

**What happens**:
1. Model loads the image
2. Analyzes it for acne
3. Draws boxes around detected lesions
4. Saves annotated image
5. Outputs JSON with all detections

**Output location**: `runs/detect/predict/acne_sample_1.jpg`

**You'll see in terminal**:
```
image 1/1: 640x640 3 comedones, 2 papules, 1 pustule
Speed: 15.2ms preprocess, 12.5ms inference, 2.1ms postprocess
Results saved to runs/detect/predict/
```

**Reading this**:
- Found 6 acne lesions total
- 3 comedones (blackheads/whiteheads)
- 2 papules (red bumps)
- 1 pustule (pus-filled)
- Took 30ms total (very fast!)

### 🔧 Step 4.2: Test YOLOv10

**Command**:
```bash
python scripts/yolov10_inference.py \
  --model runs/detect/yolov10_demo/weights/best.pt \
  --source data/samples/acne_sample_1.jpg \
  --save-json yolov10_results.json
```

**Same process**, different model.

### 🔧 Step 4.3: Visual Inspection

**Open the annotated images**:
```bash
open runs/detect/predict/acne_sample_1.jpg
```

**What to look for**:
- ✅ Boxes around actual acne lesions
- ✅ Correct classifications (comedone vs papule, etc.)
- ✅ Reasonable confidence scores (>0.4)

**Common issues with small training data**:
- May miss some lesions (low recall)
- May detect non-acne as acne (low precision)
- This improves with more training data!

---

## PHASE 5: Model Comparison {#phase-5}

### 🎯 What We're Doing
Scientifically comparing both models to see which is better.

### 🔧 Step 5.1: Benchmark Both Models

**Command**:
```bash
python scripts/compare_models.py \
  --model1 runs/detect/yolov8_demo/weights/best.pt \
  --model1-name "YOLOv8-nano" \
  --model2 runs/detect/yolov10_demo/weights/best.pt \
  --model2-name "YOLOv10-nano" \
  --test-dir data/samples/ \
  --save-report model_comparison.json \
  --save-plot model_comparison.png
```

**What this measures**:
1. **Speed**: How fast each model processes images (FPS)
2. **Accuracy**: How many lesions detected correctly
3. **Model size**: How much disk space each model uses
4. **Memory usage**: How much RAM needed

**Output**:
```
MODEL COMPARISON RESULTS
========================================
Model              Size (MB)  Speed (ms)  FPS    Detections
────────────────────────────────────────────────────────────
YOLOv8-nano       6.23       15.2        65.8   45
YOLOv10-nano      4.89       12.1        82.6   48
========================================

WINNERS:
  Fastest: YOLOv10-nano (12.1ms, 82.6 FPS)
  Smallest: YOLOv10-nano (4.89 MB)
  Most Detections: YOLOv10-nano (48 detections)
```

### 📊 Understanding the Results

**Speed (FPS - Frames Per Second)**:
- **Higher is better**
- 60+ FPS = Real-time (like video)
- 30+ FPS = Smooth
- 10+ FPS = Acceptable
- <10 FPS = Slow

**Model Size**:
- Smaller = easier to deploy (phones, web apps)
- Larger = usually more accurate

**Detections**:
- More isn't always better!
- Need to check false positives
- Quality > Quantity

### 🔧 Step 5.2: Review Comparison Chart

**Command**:
```bash
open model_comparison.png
```

**You'll see 4 charts**:
1. Inference Speed (bar chart)
2. FPS Throughput (bar chart)
3. Model Size (bar chart)
4. Total Detections (bar chart)

**For your presentation**: This visual is perfect to show students!

---

## PHASE 6: Building MVP Demo {#phase-6}

### 🎯 What We're Doing
Creating a simple, interactive demo that students can use.

### 📖 What is an MVP?

**MVP** = Minimum Viable Product

**Definition**: The simplest version that still demonstrates the core functionality.

**Our MVP will**:
- Let users upload an acne photo
- Show the AI analyzing it in real-time
- Display results with bounding boxes
- Show severity assessment

**We WON'T include** (yet):
- User accounts
- Database
- Historical tracking
- Product recommendations

### 🔧 Step 6.1: Create Simple Web Demo

I'll create a Streamlit app (easy web framework for ML demos).

**File**: `demo_app.py`

**Features**:
1. Upload image button
2. Model selection (YOLOv8 vs YOLOv10)
3. Confidence threshold slider
4. Real-time detection
5. Results display (count, severity, visualization)

### 🔧 Step 6.2: Install Streamlit

**Command**:
```bash
pip install streamlit
```

**What is Streamlit?**
- Python library for creating web apps
- No HTML/CSS/JavaScript needed!
- Perfect for ML demos

**Time**: 30 seconds

### 🔧 Step 6.3: Run the Demo

**Command**:
```bash
streamlit run demo_app.py
```

**What happens**:
1. Streamlit starts a local web server
2. Opens your browser automatically
3. Shows the acne detection interface

**You'll see**:
```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.X:8501
```

### 🎨 Demo Features Explained

**Upload Section**:
- Drag & drop or click to upload
- Supports JPG, PNG
- Shows preview

**Model Selection**:
- Dropdown: YOLOv8 or YOLOv10
- Students can compare live!

**Confidence Slider**:
- Range: 0.1 to 0.9
- Shows how it affects detections
- Lower = more detections (some false)
- Higher = fewer detections (more confident)

**Results Display**:
- Annotated image with boxes
- Count by type (comedone, papule, etc.)
- Severity level (mild/moderate/severe)
- Confidence scores

---

## PHASE 7: Presentation Prep {#phase-7}

### 🎯 What We're Doing
Creating materials to explain your project to students.

### 📝 Key Points to Cover

1. **Problem Statement** (1 slide)
   - Acne affects 80% of teenagers
   - Not everyone can see a dermatologist
   - AI can help assess severity

2. **How AI Works** (2 slides)
   - Show training data examples
   - Explain learning process
   - Visualize bounding boxes

3. **Our Approach** (2 slides)
   - YOLO object detection
   - YOLOv8 vs YOLOv10 comparison
   - Why computer vision?

4. **Results** (2 slides)
   - Show model comparison chart
   - Display example detections
   - Discuss accuracy metrics

5. **Live Demo** (5 minutes)
   - Upload sample image
   - Show detection in action
   - Change confidence threshold
   - Compare models

6. **Future Improvements** (1 slide)
   - More training data
   - Skin tone fairness
   - Mobile app
   - Product recommendations

### 🔧 Step 7.1: Create Presentation Slides

I'll create a PowerPoint template with all the key points.

**File**: `PRESENTATION.md` (Markdown format, easy to convert)

### 🔧 Step 7.2: Prepare Demo Script

**What to say during demo**:

1. **Introduction** (30 seconds)
   > "Today I'm showing an AI system that can detect and classify acne from photos. This could help people assess their skin condition without visiting a dermatologist."

2. **Upload Image** (20 seconds)
   > "Let me upload a sample image... You can see the AI is processing it..."

3. **Show Results** (40 seconds)
   > "The AI detected 15 acne lesions: 8 comedones, 5 papules, and 2 pustules. Based on this count, it classified the severity as 'moderate'. The green boxes show where each lesion was detected."

4. **Adjust Confidence** (30 seconds)
   > "If I lower the confidence threshold, it finds more lesions but some might be false positives. If I raise it, fewer but more confident detections."

5. **Compare Models** (30 seconds)
   > "Let me switch from YOLOv8 to YOLOv10... YOLOv10 is 20% faster and found 2 more lesions. This shows how AI models keep improving."

6. **Conclusion** (20 seconds)
   > "This demonstrates how computer vision can assist in medical diagnosis, making healthcare more accessible."

**Total**: ~3 minutes

---

## ✅ Final Checklist

Before your presentation, verify:

- [ ] Both models trained successfully
- [ ] Demo app runs without errors
- [ ] Sample images load correctly
- [ ] Results look reasonable
- [ ] Comparison chart generated
- [ ] Presentation slides ready
- [ ] You understand each step!

---

## 🆘 Troubleshooting Guide

### Issue: Model training is too slow
**Solution**: Use smaller batch size (`--batch 4`) or fewer epochs (`--epochs 25`)

### Issue: Out of memory error
**Solution**: Close other apps, reduce batch size, or use CPU (`--device cpu`)

### Issue: Demo app won't start
**Solution**: Check if port 8501 is free, try `streamlit run demo_app.py --server.port 8502`

### Issue: Poor detection accuracy
**Expected with small dataset!** Explain this is a demo and would improve with:
- More training data (2,692 full images)
- Longer training (200 epochs)
- Better labels

---

## 📊 Expected Timeline

| Phase | Task | Time |
|-------|------|------|
| 1 | Environment setup | 15 min |
| 2 | Dataset prep | 10 min |
| 3 | Train both models | 30 min |
| 4 | Test & evaluate | 10 min |
| 5 | Compare models | 5 min |
| 6 | Build MVP demo | 20 min |
| 7 | Prep presentation | 30 min |
| **Total** | | **~2 hours** |

With full dataset: Add 3-4 hours for training.

---

## 🎓 Learning Outcomes

By completing this walkthrough, you'll understand:

1. **Machine Learning Pipeline**
   - Data preparation
   - Model training
   - Evaluation
   - Deployment

2. **Computer Vision Concepts**
   - Object detection
   - Bounding boxes
   - Confidence scores
   - Model architectures

3. **Practical AI Development**
   - Setting up environments
   - Using pre-trained models
   - Benchmarking
   - Creating demos

4. **Scientific Comparison**
   - Metrics that matter
   - Speed vs accuracy tradeoffs
   - Model evolution (v8 → v10)

---

**Ready to start?** Let's begin with Phase 1!
