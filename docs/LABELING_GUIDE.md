# Acne Image Labeling Guide

## Overview

This guide explains how to automatically label your acne images using the Roboflow ensemble detector to generate CSV files like [data/skin_features.csv](../data/skin_features.csv).

## What Gets Labeled

The labeling script generates the following features for each image:

### 📊 **Detection Features**
- `acne_count`: Total number of acne lesions detected
- `avg_acne_width`: Average width of bounding boxes (pixels)
- `avg_acne_height`: Average height of bounding boxes (pixels)
- `avg_acne_area`: Average area of lesions (pixels²)

### 🔬 **Acne Type Breakdown**
- `papules_count`: Inflammatory lesions (red bumps)
- `pustules_count`: Pus-filled lesions
- `comedone_count`: Blackheads and whiteheads
- `nodules_count`: Large, deep lesions

### 🎨 **Color Features**
- `avg_redness`: Average redness of detected lesion regions
- `global_redness`: Overall redness of entire image

### 🏷️ **Classification**
- `skin_disease_label`: Overall classification (e.g., "Acne and Rosacea Photos")
- `skin_disease_confidence`: Confidence score (0-1)
- `acne_detected`: Binary flag (1 = acne found, 0 = no acne)

## How It Works

### **Step 1: Get Roboflow API Key**

1. Go to [Roboflow](https://roboflow.com/)
2. Sign up or log in
3. Click your profile → "Account Settings"
4. Copy your API key

### **Step 2: Run the Labeling Script**

```bash
# Label all datasets (acne + rosacea)
python scripts/generate_labels.py --api-key YOUR_API_KEY --all

# Or label a specific directory
python scripts/generate_labels.py \
  --api-key YOUR_API_KEY \
  --input data/raw/acne \
  --output data/labels/acne_labels.csv
```

### **Step 3: Review the Output**

The script will create CSV files in `data/labels/` with the same structure as your existing [data/skin_features.csv](../data/skin_features.csv).

```
data/labels/
├── acne_labels.csv       (2,690 images labeled)
└── rosacea_labels.csv    (282 images labeled)
```

## Example Output

```csv
ID,filename,acne_count,avg_acne_width,avg_acne_height,avg_acne_area,papules_count,pustules_count,comedone_count,nodules_count,avg_redness,global_redness,skin_disease_label,skin_disease_confidence,skin_classification_labels,acne_detected,result
1,acne-image-001.jpg,5,28.2,21.4,626.0,3,0,2,0,0.14157,0.10478,Acne and Rosacea Photos,0.942,NULL,1,NaN
2,clear-skin-002.jpg,0,0,0,0,0,0,0,0,0,0.08134,Acne and Rosacea Photos,0.513,NULL,0,NaN
```

## Behind the Scenes: Ensemble Logic

The script uses [src/ensemble_detector.py](../src/ensemble_detector.py) which runs **3 Roboflow models**:

```
Your Image
    │
    ├──► acnedet-v1          Primary detector (draws boxes)
    ├──► skin_disease_ak      Classifier (labels condition)
    └──► skn-1               Fallback detector
         │
         ▼
    Ensemble Logic
         │
         ├─ If acnedet-v1 confident → use it
         ├─ If classifier says "acne" but no detections → use skn-1
         └─ Combine results with confidence weighting
         │
         ▼
    Final Labels
```

### **Why Ensemble?**

| Model | Strengths | Weaknesses |
|-------|-----------|------------|
| **acnedet-v1** | High precision on clear headshots | Fails on blurry/distant images |
| **skin_disease_ak** | Good at overall classification | Sometimes returns wrong labels |
| **skn-1** | More robust to poor quality | Lower precision |

By combining them, you get:
- ✅ High accuracy on good images (acnedet-v1)
- ✅ Fallback for challenging images (skn-1)
- ✅ Validation via classification (skin_disease_ak)

## Customizing the Labeling

### **Adjust Confidence Thresholds**

Edit the script to change detection sensitivity:

```python
detector = AcneEnsembleDetector(
    api_key=api_key,
    confidence_threshold=0.6,  # Default: 0.4 (lower = more detections)
    classification_threshold=0.90  # Default: 0.85
)
```

### **Filter by Acne Type**

If you only want certain types:

```python
# In count_acne_types(), filter out unwanted types
if acne_type == 'nodules':
    continue  # Skip nodules
```

### **Add Custom Features**

You can extend the script to add more features:

```python
def calculate_skin_tone(image: np.ndarray) -> float:
    """Calculate average skin tone"""
    # Your custom logic
    return tone_score

# Then add to row:
row['skin_tone'] = calculate_skin_tone(image)
```

## Comparing with Existing Labels

If you already have [data/skin_features.csv](../data/skin_features.csv), you can compare:

```python
import pandas as pd

# Load both
existing = pd.read_csv('data/skin_features.csv')
new = pd.read_csv('data/labels/acne_labels.csv')

# Compare acne counts
comparison = existing.merge(new, on='filename', suffixes=('_old', '_new'))
comparison['count_diff'] = comparison['acne_count_new'] - comparison['acne_count_old']

print(comparison[['filename', 'acne_count_old', 'acne_count_new', 'count_diff']])
```

## Tips for Best Results

### ✅ **Do:**
- Use high-resolution images (min 640x640)
- Ensure good lighting in photos
- Run on a sample first to verify results
- Review low-confidence detections manually

### ❌ **Don't:**
- Label images with faces partially cut off
- Use images with heavy filters/makeup
- Process images smaller than 300x300 pixels
- Trust very low confidence scores (<0.3)

## Processing Time

Expected processing time:
- **2,690 images (acne dataset)**: ~45-60 minutes
- **282 images (rosacea dataset)**: ~5-7 minutes

The script shows a progress bar with estimated time remaining.

## Troubleshooting

### **Error: "No module named 'roboflow'"**
```bash
pip install roboflow
```

### **Error: "Failed to load models"**
- Check your API key is correct
- Verify internet connection
- Ensure Roboflow account has access to models

### **Warning: "Many false positives"**
Increase the confidence threshold:
```bash
# Edit scripts/generate_labels.py, line ~193
confidence_threshold=0.6  # Increase from 0.4
```

### **Issue: "Processing is too slow"**
- Process in batches (split dataset)
- Use GPU if available
- Reduce image resolution before processing

## Next Steps

After labeling:

1. **Review the CSV** - Check for anomalies or errors
2. **Visualize distributions** - Plot acne counts, types
3. **Train custom model** - Use labels for supervised learning
4. **Build severity classifier** - Predict mild/moderate/severe
5. **Create recommendation engine** - Map labels to product suggestions

## Example Notebook

Check [notebooks/analyze_labels.ipynb](../notebooks/analyze_labels.ipynb) for:
- Loading and exploring label CSVs
- Visualizing acne type distributions
- Analyzing detection confidence
- Comparing ensemble model performance

## Related Files

- [src/ensemble_detector.py](../src/ensemble_detector.py) - Core detection logic
- [scripts/generate_labels.py](../scripts/generate_labels.py) - Labeling script
- [data/skin_features.csv](../data/skin_features.csv) - Example existing labels
- [config/default.yaml](../config/default.yaml) - Configuration settings

## Questions?

- Read [QUICK_START_ACNE.md](../QUICK_START_ACNE.md) for ensemble detector tips
- Check [ACNE_DETECTION_PIVOT.md](../ACNE_DETECTION_PIVOT.md) for strategy details
- Review ensemble detector code in [src/ensemble_detector.py](../src/ensemble_detector.py)
