# Quick Start Guide - Acne Detection Pivot

## TL;DR - What Changed

**Before**: General skin lesion detection (HAM10000, ISIC)
**After**: Focused acne detection with ensemble Roboflow models

## Datasets: Keep vs Remove

### ✅ KEEP (Acne-Focused)

| Dataset | Priority | Why |
|---------|----------|-----|
| **Acne Dataset** (Kaggle) | 🔴 HIGH | 1,800 acne images - your core dataset |
| **Acne-Wrinkles-Spots** | 🟡 MEDIUM | Has acne category, good supplement |
| **Skin Disease** (filtered) | 🟡 MEDIUM | Filter for acne/rosacea cases only |
| **FitzPatrick17k** (filtered) | 🟢 LOW | For diversity/bias testing |

### ❌ REMOVE (Not Acne)

- ❌ **HAM10000** - Melanoma/cancer focused
- ❌ **ISIC Archive** - Skin cancer focused

## Quick Setup

### 1. Download Acne Datasets Only

```bash
# Install Kaggle API first (one-time setup)
mkdir -p ~/.kaggle
# Download kaggle.json from https://www.kaggle.com/settings/account
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download acne datasets
python scripts/download_acne_datasets.py --all

# Filter broad datasets for acne only
python scripts/download_acne_datasets.py --filter-acne

# Create sample images
python scripts/download_acne_datasets.py --create-samples
```

### 2. Use the Ensemble Detector

```python
from src.ensemble_detector import AcneEnsembleDetector

# Initialize with your Roboflow API key
detector = AcneEnsembleDetector(api_key="YOUR_API_KEY")

# Detect acne
result = detector.detect("path/to/image.jpg")

print(f"Acne count: {result.count}")
print(f"Confidence: {result.confidence_level}")
print(f"Model used: {result.primary_model}")

# Get severity assessment
severity = detector.assess_severity(result)
print(f"Severity: {severity['severity']}")  # mild, moderate, severe
```

## Tips for Your Team Member

### 1. Handling the "Viral infections" Issue

**Problem**: skin_disease_ak sometimes returns wrong labels

**Solution**: Whitelist valid labels only

```python
VALID_ACNE_LABELS = {
    "Acne and Rosacea Photos",
    "Acne Vulgaris",
    "Comedones",
    "Papules",
    "Pustules"
}

# In ensemble logic:
if classifier_label not in VALID_ACNE_LABELS:
    # Ignore this classification
    pass
```

### 2. Improving Detection on Poor Quality Images

**Problem**: acnedet-v1 fails on blurry/distant images

**Solution**: Preprocess before detection

```python
from src.ensemble_detector import ImagePreprocessor, ImageQualityAssessor

assessor = ImageQualityAssessor()
preprocessor = ImagePreprocessor()

# Assess quality
quality = assessor.assess(image)

# Enhance if needed
if quality['quality'] != 'high':
    image = preprocessor.enhance(image, quality['quality'])

# Then run detection
result = detector.detect(image)
```

### 3. When to Use Each Model

| Scenario | Primary | Fallback | Why |
|----------|---------|----------|-----|
| Crisp headshot | acnedet-v1 | - | Best precision |
| Blurry image | skn-1 | acnedet-v1 | More robust |
| No detections but classifier says "acne" | skn-1 | - | Catch what acnedet missed |
| Classifier disagrees | Use with caution | - | Add warning |

### 4. Fine-Tuning Your Models

**Option A: Fine-tune on Roboflow (Easiest)**

1. Upload your acne dataset to Roboflow
2. Combine with existing acnedet-v1 dataset
3. Retrain with your specific images
4. Deploy updated model

**Option B: Train Custom YOLOv8**

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(
    data='config/acne_data.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```

### 5. Testing for Fairness (Critical!)

Use FitzPatrick17k to test across skin tones:

```python
# Test on different skin types
results_by_skin_tone = {}

for skin_type in ['I', 'II', 'III', 'IV', 'V', 'VI']:
    images = load_fitzpatrick_images(skin_type, condition='acne')

    for img in images:
        result = detector.detect(img)
        # Track performance
        results_by_skin_tone[skin_type].append(result)

# Check for bias
variance = calculate_variance(results_by_skin_tone)
if variance > 0.10:  # 10% threshold
    print("⚠️ Warning: Model may be biased across skin tones")
```

## Mapping to Recommendations

Once you have detection results:

```python
def get_product_recommendations(result, severity):
    """
    Map detection results to product recommendations
    """
    recommendations = []

    if severity['severity'] == 'mild':
        recommendations = [
            {'product': 'Gentle Salicylic Acid Cleanser', 'priority': 'high'},
            {'product': 'Non-comedogenic Moisturizer', 'priority': 'high'},
            {'product': 'BHA Toner', 'priority': 'medium'}
        ]

    elif severity['severity'] == 'moderate':
        recommendations = [
            {'product': 'Benzoyl Peroxide 2.5% Gel', 'priority': 'high'},
            {'product': 'Niacinamide Serum', 'priority': 'high'},
            {'product': 'Oil-free Sunscreen SPF 30+', 'priority': 'medium'}
        ]

    elif severity['severity'] == 'severe':
        recommendations = [
            {'recommendation': 'Consult a dermatologist', 'priority': 'urgent'},
            {'product': 'Gentle Cleanser', 'priority': 'high'},
            {'note': 'Prescription medication may be needed'}
        ]

    # Filter by acne type
    if 'comedone' in severity['type_breakdown']:
        recommendations.append({
            'product': 'Retinoid Cream (OTC)',
            'priority': 'medium'
        })

    if 'pustule' in severity['type_breakdown']:
        recommendations.append({
            'product': 'Spot Treatment with Benzoyl Peroxide',
            'priority': 'high'
        })

    return recommendations
```

## Updated Timeline

### Week 1: Data (THIS WEEK)
- [x] Remove HAM10000/ISIC from pipeline
- [ ] Download acne-specific datasets
- [ ] Filter and organize acne images
- [ ] Create unified dataset
- [ ] Set up DVC tracking

### Week 2: Model (NEXT WEEK)
- [ ] Implement ensemble detector (code provided)
- [ ] Test on diverse images (FitzPatrick17k)
- [ ] Fine-tune best model on your data
- [ ] Measure fairness metrics
- [ ] Document model behavior

### Week 3: Integration
- [ ] Connect ensemble to FastAPI
- [ ] Build frontend upload interface
- [ ] Map detections → severity → recommendations
- [ ] Test end-to-end pipeline

### Week 4: QA
- [ ] User testing
- [ ] Edge case handling
- [ ] Performance optimization
- [ ] Final documentation

## Common Issues & Solutions

### Issue: "No module named 'roboflow'"

```bash
pip install roboflow
```

### Issue: "Predictions vary wildly"

**Solution**: Log image quality and model agreement

```python
# Track consistency
if result.confidence_level == 'low':
    # Ask user to retake photo with better lighting
    return {'error': 'Please retake photo in better lighting'}
```

### Issue: "Too many false positives"

**Solution**: Increase confidence threshold

```python
detector = AcneEnsembleDetector(
    api_key=api_key,
    confidence_threshold=0.6,  # Increase from 0.4
    classification_threshold=0.90  # Increase from 0.85
)
```

### Issue: "Model is slow"

**Solution**:
1. Use smaller YOLOv8 model (yolov8n instead of yolov8m)
2. Reduce image resolution
3. Cache results for same image

## Key Files Created

| File | Purpose |
|------|---------|
| [ACNE_DETECTION_PIVOT.md](ACNE_DETECTION_PIVOT.md) | Complete strategy guide |
| [scripts/download_acne_datasets.py](scripts/download_acne_datasets.py) | Download acne datasets |
| [src/ensemble_detector.py](src/ensemble_detector.py) | Ensemble detection logic |
| [SETUP.md](SETUP.md) | General setup instructions |
| [data/README.md](data/README.md) | Data management guide |

## Questions?

Check these docs in order:
1. **QUICK_START_ACNE.md** (this file) - Quick answers
2. **ACNE_DETECTION_PIVOT.md** - Detailed strategy
3. **src/ensemble_detector.py** - Implementation reference
4. **data/README.md** - Data management help

## Next Steps for Your Team Member

1. **Today**: Download acne datasets, remove HAM10000/ISIC references
2. **This Week**: Test ensemble_detector.py with their Roboflow models
3. **Next Week**: Fine-tune on combined acne dataset
4. **Week 3**: Integrate into FastAPI + frontend

Good luck! 🚀
