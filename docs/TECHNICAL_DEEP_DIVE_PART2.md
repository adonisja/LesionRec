# Deep Dive Technical Learning - Part 2
**Advanced Topics: Loss Functions, Metrics, Transfer Learning, Augmentation, and GPU Acceleration**

*Continuation of DEEP_DIVE_TECHNICAL_LEARNING.md*

---

## 8. Loss Functions Explained {#loss-functions}

### What is a Loss Function?

**Definition**: A mathematical function that measures how wrong your model's predictions are.

```
Perfect Prediction → Loss = 0
Terrible Prediction → Loss = large number

Training Goal: Minimize Loss
```

### Binary Cross-Entropy (BCE) Loss

**Used for**: Binary classification (yes/no, acne/no-acne)

**Formula**:
```
L = -1/N × Σ[y * log(p) + (1-y) * log(1-p)]

Where:
  y = true label (0 or 1)
  p = predicted probability
  N = number of samples
```

**Example**:
```python
# Ground truth: Acne present
y = 1

# Model predictions:
p1 = 0.95  # Confident and correct
p2 = 0.55  # Uncertain but correct
p3 = 0.05  # Confident but WRONG

# Calculate losses:
loss1 = -(1 * log(0.95) + 0 * log(0.05)) = 0.051  # Low loss ✓
loss2 = -(1 * log(0.55) + 0 * log(0.45)) = 0.598  # Medium loss
loss3 = -(1 * log(0.05) + 0 * log(0.95)) = 2.996  # High loss ✗

Observation:
  • Correct + confident → Low loss
  • Wrong + confident → VERY high loss (penalized heavily!)
```

**Intuition**:
```
If y=1 (positive class):
  p close to 1 → log(p) ≈ 0 → low loss
  p close to 0 → log(p) → -∞ → high loss

If y=0 (negative class):
  p close to 0 → log(1-p) ≈ 0 → low loss
  p close to 1 → log(1-p) → -∞ → high loss
```

### IoU (Intersection over Union) Loss

**Used for**: Bounding box regression

**Definition**:
```
IoU = Area of Overlap / Area of Union

       ┌──────────┐
       │ Pred Box │
       │    ┌─────┼────┐
       │    │/////│    │
       └────┼─────┘    │ Ground Truth
            │   Overlap│
            └──────────┘

IoU = (Overlap Area) / (Total Area Covered)

Perfect match: IoU = 1.0
No overlap: IoU = 0.0
```

**Calculation**:
```python
def calculate_iou(box1, box2):
    """
    box1, box2: [x1, y1, x2, y2]
    """
    # Calculate intersection
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    intersection = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    # Calculate union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    iou = intersection / union
    return iou

# Example:
pred_box = [10, 10, 50, 50]  # 40×40 = 1600 pixels
true_box = [20, 20, 60, 60]  # 40×40 = 1600 pixels

# Overlap: [20,20] to [50,50] = 30×30 = 900 pixels
# Union: 1600 + 1600 - 900 = 2300 pixels
# IoU: 900 / 2300 = 0.391
```

**IoU Loss**:
```
IoU_Loss = 1 - IoU

Perfect match: Loss = 0
No overlap: Loss = 1
```

### CIoU (Complete IoU) Loss

**Problem with IoU**: Doesn't account for:
1. Distance between box centers
2. Aspect ratio difference

**CIoU Solution**: Adds penalty terms

```
CIoU_Loss = 1 - IoU + (distance_penalty + aspect_ratio_penalty)

Distance Penalty:
  ρ²(b, b_gt) / c²

  Where:
    ρ = Euclidean distance between centers
    c = diagonal length of smallest enclosing box

Aspect Ratio Penalty:
  α × v

  Where:
    v = (4/π²) × (arctan(w_gt/h_gt) - arctan(w/h))²
    α = v / (1 - IoU + v)
```

**Why CIoU is Better**:
```
Scenario 1: Same IoU, Different Positions
┌─────┐         ┌─────┐
│ GT  │         │ GT  │
│ ┌─┐ │         │     │
│ └─┘ │         └─┬─┬─┘
└─────┘           └─┘
Pred close        Pred far

IoU: Same for both
CIoU: Lower loss for left (centers closer)
```

```
Scenario 2: Same IoU, Different Shapes
┌───────┐       ┌─────┐
│  GT   │       │ GT  │
│ ┌───┐ │       │ ┌─┐ │
│ └───┘ │       │ │ │ │
└───────┘       │ └─┘ │
Pred: wide      └─────┘
                Pred: tall

IoU: Same for both
CIoU: Lower loss for matching aspect ratio
```

### YOLOv8 Total Loss

**Combined Loss Function**:
```
Total_Loss = λ₁×box_loss + λ₂×cls_loss + λ₃×dfl_loss

Where:
  box_loss: Bounding box regression (CIoU)
  cls_loss: Classification (BCE)
  dfl_loss: Distribution Focal Loss
  λ₁, λ₂, λ₃: Weight coefficients
```

**Breaking Down Each Component**:

**1. Box Loss (CIoU)**:
```python
# For each predicted box:
ciou = calculate_ciou(pred_box, true_box)
box_loss = 1 - ciou

# Example from your training:
epoch   1/100: box_loss = 1.1234
  Interpretation: Boxes are overlapping ~12% (1 - 0.12 = 0.88)

epoch 100/100: box_loss = 0.3456
  Interpretation: Boxes are overlapping ~65% (much better!)
```

**2. Classification Loss (BCE)**:
```python
# For each detection:
cls_loss = -[y * log(p) + (1-y) * log(1-p)]

# With 4 acne classes:
true_class = [0, 1, 0, 0]  # Papule
pred_class = [0.1, 0.7, 0.15, 0.05]

cls_loss = -(0*log(0.1) + 1*log(0.7) + 0*log(0.15) + 0*log(0.05))
         = -log(0.7)
         = 0.357

# Example from your training:
epoch   1/100: cls_loss = 1.6789
  Interpretation: Model very uncertain about classes

epoch 100/100: cls_loss = 0.2345
  Interpretation: Model confident and accurate!
```

**3. DFL Loss (Distribution Focal Loss)**:
```
Traditional: Predict exact box coordinates
DFL: Predict probability distribution over coordinates

Instead of:
  x = 0.45 (single value)

DFL predicts:
  P(x=0.44) = 0.2
  P(x=0.45) = 0.6  ← Peak
  P(x=0.46) = 0.2

Benefits:
  • Captures uncertainty
  • More robust to ambiguous boundaries
  • Better gradient flow

Loss:
  dfl_loss = -Σ(y × log(p))

  Where y is one-hot encoded true distribution
```

### Watching Loss During Training

**What Good Training Looks Like**:
```
Epoch    box_loss  cls_loss  dfl_loss
   1     1.1234    1.6789    1.2345   ← Start high
  10     0.8234    1.2789    0.9876
  25     0.5678    0.8123    0.7234
  50     0.4123    0.5234    0.5678   ← Steady decrease
  75     0.3567    0.3456    0.4567
 100     0.3234    0.2345    0.4234   ← Converged

All losses decreasing smoothly ✓
```

**What Bad Training Looks Like**:
```
Epoch    box_loss  cls_loss  dfl_loss
   1     1.1234    1.6789    1.2345
  10     0.9234    1.4789    1.0876
  25     1.2678    2.1123    1.5234   ← Increasing! ✗
  50     0.7123    1.8234    1.2678
  75     1.5567    0.9456    2.1567   ← Unstable ✗
 100     0.8234    1.5345    0.9234

Problems:
  • Losses jumping around → Learning rate too high
  • Losses increasing → Model diverging
  • Losses stuck → Learning rate too low or bad data
```

---

## 9. Evaluation Metrics {#evaluation-metrics}

### Confusion Matrix

**For Binary Classification** (Acne vs No Acne):

```
                    Predicted
                 Acne    Normal
Actual  Acne      TP       FN
        Normal    FP       TN

TP (True Positive): Correctly detected acne
FP (False Positive): Detected acne but none exists (false alarm)
FN (False Negative): Missed acne (dangerous!)
TN (True Negative): Correctly identified normal skin
```

**Example**:
```
You test on 100 images:
  50 with acne
  50 normal

Results:
  TP = 45  (Detected 45 out of 50 acne cases)
  FP = 5   (False alarm on 5 normal images)
  FN = 5   (Missed 5 acne cases)
  TN = 45  (Correctly identified 45 normal images)

              Predicted
           Acne  Normal
Actual Acne  45     5
       Normal 5    45
```

### Precision

**Definition**: Of all the detections, how many were correct?

```
Precision = TP / (TP + FP)
          = True Positives / All Positive Predictions

Example:
  TP = 45, FP = 5
  Precision = 45 / (45 + 5) = 45/50 = 0.90 (90%)

Interpretation:
  "When my model says 'acne', it's correct 90% of the time"
```

**High Precision**:
- Few false alarms
- When model detects acne, you can trust it
- Good for: Medical diagnosis (avoid unnecessary treatment)

**Low Precision**:
- Many false alarms
- Model is "trigger-happy"

### Recall (Sensitivity)

**Definition**: Of all the actual acne cases, how many did we detect?

```
Recall = TP / (TP + FN)
       = True Positives / All Actual Positives

Example:
  TP = 45, FN = 5
  Recall = 45 / (45 + 5) = 45/50 = 0.90 (90%)

Interpretation:
  "My model catches 90% of all acne cases"
```

**High Recall**:
- Catches most acne
- Few missed cases
- Good for: Cancer detection (can't afford to miss!)

**Low Recall**:
- Misses many cases
- Dangerous in medical applications

### Precision vs Recall Trade-off

```
Confidence Threshold
         ↓
     Model Output
         ↓
If probability > threshold → Predict "Acne"
If probability ≤ threshold → Predict "Normal"

Example:
  True label: Acne
  Model output: 0.65

Scenario 1: threshold = 0.5
  0.65 > 0.5 → Predict "Acne" ✓

Scenario 2: threshold = 0.8
  0.65 < 0.8 → Predict "Normal" ✗ (Missed!)
```

**Low Threshold** (e.g., 0.3):
```
More detections
  → Higher Recall (catch everything)
  → Lower Precision (many false alarms)

Example:
  Detect anything with >30% confidence

  Image 1: 95% → Acne ✓
  Image 2: 65% → Acne ✓
  Image 3: 35% → Acne (but actually normal) ✗
  Image 4: 31% → Acne (but actually normal) ✗

  Caught all real acne, but many false alarms
```

**High Threshold** (e.g., 0.8):
```
Fewer detections
  → Lower Recall (miss borderline cases)
  → Higher Precision (only confident detections)

Example:
  Only detect if >80% confident

  Image 1: 95% → Acne ✓
  Image 2: 65% → (skipped, might be acne) ✗
  Image 3: 35% → (skipped)
  Image 4: 31% → (skipped)

  No false alarms, but missed some real acne
```

**Visualization**:
```
Precision
   ↑
 1 │     ╱╲
   │    ╱  ╲___
   │   ╱       ╲___
   │  ╱            ╲___
   │ ╱                 ╲___
 0 └───────────────────────→ Recall
   0                        1

Sweet spot: Balance both (F1 score)
```

### F1 Score

**Definition**: Harmonic mean of Precision and Recall

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)

Example:
  Precision = 0.90
  Recall = 0.90
  F1 = 2 × (0.90 × 0.90) / (0.90 + 0.90) = 0.90

Example 2 (Imbalanced):
  Precision = 0.95 (very few false alarms)
  Recall = 0.50 (miss half the cases!)
  F1 = 2 × (0.95 × 0.50) / (0.95 + 0.50) = 0.655

F1 penalizes imbalance!
```

**Why Harmonic Mean?**
```
Arithmetic mean would be: (0.95 + 0.50) / 2 = 0.725
Harmonic mean: 0.655

Harmonic mean is always ≤ arithmetic mean
It's dominated by the smaller value
→ Forces you to care about both metrics!
```

### mAP (mean Average Precision)

**Most Important Metric for Object Detection**

**Step 1**: Calculate Precision and Recall at different thresholds

```python
# For confidence thresholds: 0.1, 0.2, ..., 0.9, 1.0
thresholds = [0.1, 0.2, 0.3, ..., 1.0]

for threshold in thresholds:
    predictions = model.predict(images, conf=threshold)
    precision, recall = calculate_metrics(predictions, ground_truth)
    store(precision, recall)
```

**Step 2**: Plot Precision-Recall Curve

```
Precision
   ↑
 1 │●
   │ ●
   │  ●●
   │    ●●
   │      ●●●
   │         ●●●
   │            ●●●●
 0 └────────────────●──→ Recall
   0                   1

Each ● = one confidence threshold
```

**Step 3**: Calculate Area Under Curve

```
Average Precision (AP) = Area under Precision-Recall curve

AP close to 1.0 → Excellent detector
AP close to 0.0 → Poor detector
```

**Step 4**: Calculate mAP (mean over all classes)

```
AP_comedone = 0.85
AP_papule = 0.78
AP_pustule = 0.82
AP_nodule = 0.75

mAP = (0.85 + 0.78 + 0.82 + 0.75) / 4 = 0.80 (80%)
```

### mAP@50 vs mAP@50-95

**mAP@50**: IoU threshold = 0.5
```
Only count as correct if:
  IoU(predicted_box, true_box) ≥ 0.5

┌────────┐
│  True  │
│  ┌─────┼──┐
│  │ 50% │  │ Predicted
└──┼─────┘  │
   └────────┘

This counts as correct (IoU ≥ 0.5)
```

**mAP@50-95**: Average over IoU thresholds 0.5, 0.55, 0.60, ..., 0.95
```
More strict!

Must have good IoU at multiple thresholds:
  IoU ≥ 0.50 ✓
  IoU ≥ 0.55 ✓
  IoU ≥ 0.60 ✓
  ...
  IoU ≥ 0.95 ✗ (very hard!)

Only perfect boxes score well here
```

**Comparison**:
```
Model A:
  mAP@50 = 0.85 (good at loose matching)
  mAP@50-95 = 0.45 (poor at tight matching)
  → Detects acne but boxes are sloppy

Model B:
  mAP@50 = 0.82 (slightly lower)
  mAP@50-95 = 0.70 (much better!)
  → Very precise bounding boxes

Model B is better for medical use!
```

### Your Training Metrics

**YOLOv8 Reports**:
```
metrics/precision
metrics/recall
metrics/mAP50
metrics/mAP50-95

Example output:
  Precision: 0.856
  Recall: 0.812
  mAP@50: 0.834
  mAP@50-95: 0.623

Interpretation:
  • 85.6% of detections are correct (precision)
  • Catches 81.2% of all acne (recall)
  • 83.4% average precision at IoU=0.5
  • 62.3% average precision across strict IoUs
```

**What to Aim For**:
```
Medical Application:
  mAP@50 > 0.80 (Good detection)
  Recall > 0.85 (Catch most cases)
  Precision > 0.75 (Minimize false alarms)

Your current goal:
  mAP@50 > 0.75 (decent for demo/MVP)
  Improving with more training epochs
```

---

## 10. Transfer Learning in Depth {#transfer-learning}

### What is Transfer Learning?

**Traditional Approach** (Training from scratch):
```
┌────────────────────┐
│ Random Initialization
│ (Know nothing)     │
└─────────┬──────────┘
          │ Train on 2,690 images
          │ (Need millions for good results!)
          ↓
    Poor Performance
    (Not enough data)
```

**Transfer Learning Approach**:
```
┌─────────────────────┐
│ Pre-trained Model   │
│ (Trained on millions│
│  of COCO images)    │
│                     │
│ Learned:            │
│  • Edges            │
│  • Textures         │
│  • Shapes           │
│  • Objects          │
└──────────┬──────────┘
           │ Fine-tune on 2,690 acne images
           │ (Adapt general knowledge to acne)
           ↓
    Excellent Performance
    (Leverage prior knowledge!)
```

### Layer-by-Layer Understanding

**Pre-trained YOLOv8 Layers**:

```
Layer 1-5 (Early Layers):
┌───────────────────────┐
│ Detect Basic Features │
│  • Edges              │
│  • Colors             │
│  • Textures           │
│  • Gradients          │
└───────────┬───────────┘
            │
            │ These are UNIVERSAL
            │ (Same for all images)
            │ → FREEZE these layers
            ↓

Layer 6-15 (Middle Layers):
┌───────────────────────┐
│ Detect Object Parts   │
│  • Circles            │
│  • Lines              │
│  • Corners            │
│  • Patterns           │
└───────────┬───────────┘
            │
            │ Somewhat general
            │ → FINE-TUNE these layers
            │    (small learning rate)
            ↓

Layer 16-25 (Deep Layers):
┌───────────────────────┐
│ Detect Complex Objects│
│  • People             │
│  • Cars               │
│  • Animals            │
│  • ...                │
└───────────┬───────────┘
            │
            │ Task-specific
            │ → RETRAIN these layers
            │    (normal learning rate)
            ↓

Output Layer:
┌───────────────────────┐
│ Class Predictions     │
│  COCO: 80 classes     │
│    ↓                  │
│  ACNE: 4 classes      │
│  • Comedone           │
│  • Papule             │
│  • Pustule            │
│  • Nodule             │
└───────────────────────┘
   → REPLACE and RETRAIN
```

### Feature Reusability

**Why Transfer Learning Works**:

```
COCO Dataset (pre-training):
  "person" class → Detects faces, skin, body parts

Acne Dataset (fine-tuning):
  Comedone → Appears on skin (detected by person class!)

The model already knows:
  ✓ How to detect skin textures
  ✓ How to find circular objects (faces → acne lesions)
  ✓ How to handle varying lighting
  ✓ How to deal with different skin tones

Just needs to learn:
  ✗ Specific acne lesion types
  ✗ Fine-grained differences (comedone vs papule)
  ✗ Medical-specific features (inflammation, redness)
```

**Visualization**:
```
Pre-trained Feature Maps (from faces):

Layer 5 Output:
┌────────────┐
│  Edges     │
│  Detected  │  → Useful for acne edges!
└────────────┘

Layer 10 Output:
┌────────────┐
│  Skin      │
│  Texture   │  → Useful for detecting skin!
└────────────┘

Layer 15 Output:
┌────────────┐
│  Circular  │
│  Patterns  │  → Useful for round lesions!
└────────────┘

Layer 20 Output:
┌────────────┐
│  Faces     │
│  Detected  │  → Need to adapt to acne lesions
└────────────┘
```

### Training Strategy

**YOLOv8 Transfer Learning**:

```python
from ultralytics import YOLO

# Load pre-trained model
model = YOLO('yolov8m.pt')  # Pre-trained on COCO

# Under the hood:
#   - Loads 25M parameters trained on 118k images
#   - Replaces output layer (80 classes → 4 classes)
#   - Keeps all other layers intact

# Train on acne data
model.train(
    data='config/yolo_acne.yaml',
    epochs=100,
    imgsz=640,

    # Transfer learning magic:
    freeze=10,  # Freeze first 10 layers
                # (they already know edges/textures)

    lr0=0.01,   # Initial learning rate
                # (smaller than training from scratch)

    warmup_epochs=3  # Gradual warm-up
                     # (stabilize transferred weights)
)
```

**What Happens During Training**:

```
Epoch 1-10: Quick Adaptation
------------------------------------------------------------
  Early layers (frozen): Unchanged
  Middle layers: Small adjustments
  Deep layers: Rapid learning
  Output layer: Complete retraining

  Loss drops fast! (leveraging prior knowledge)

Epoch 11-50: Fine-Tuning
------------------------------------------------------------
  All layers: Gradual refinement
  Model adapts general features to acne-specific

  Loss decreases steadily

Epoch 51-100: Polishing
------------------------------------------------------------
  Weights converge
  Model masters acne detection

  Loss reaches minimum
```

### Data Efficiency

**Comparison**:

```
Training from Scratch:
  Data needed: 10,000+ images per class
  Training time: Weeks
  Final accuracy: 60-70% (with your 2,690 images)

Transfer Learning:
  Data needed: 500+ images per class  ← You have this!
  Training time: Hours
  Final accuracy: 80-90% (same 2,690 images)

Improvement: ~20-30% accuracy boost!
```

**Why It's More Data Efficient**:
```
From Scratch:
  Model learns: "What is an edge?"
  Requires: 1000s of examples

Transfer Learning:
  Model knows: "What is an edge"
  Learns: "Which edges indicate acne?"
  Requires: 100s of examples

Lower-level knowledge is already there!
```

---

## 11. Data Augmentation {#data-augmentation}

### Why Augment Data?

**Problem**: Limited training data (2,690 images)

**Solution**: Create variations artificially

```
Original Image:
┌────────────┐
│   Acne     │
│     ●      │
│            │
└────────────┘
     ↓
Augmentations:
┌────────────┬────────────┬────────────┬────────────┐
│  Flipped   │  Rotated   │  Brighter  │  Zoomed    │
│     ●      │      ●     │     ●      │     ●●     │
│            │    /       │   ░░░      │     ●●     │
└────────────┴────────────┴────────────┴────────────┘

1 image → 5+ variations
Model sees "different" images, learns to generalize!
```

### Augmentation Techniques

**1. Horizontal Flip**:
```python
augmented = cv2.flip(image, 1)

Before:           After:
┌────┐            ┌────┐
│  ● │     →      │ ●  │
└────┘            └────┘

Medical relevance:
  Acne on left cheek = same as right cheek
  Perfectly valid augmentation!
```

**2. Rotation**:
```python
augmented = rotate(image, angle=15)

Before:           After:
┌────┐            ┌────┐
│  ● │     →      │   ●│
└────┘            └────┘

Limits:
  • Keep rotation small (±15°)
  • Larger rotations unrealistic (faces aren't upside down!)
```

**3. Brightness/Contrast**:
```python
augmented = adjust_brightness(image, factor=1.2)

Before:           After:
┌────┐            ┌────┐
│  ● │     →      │  ● │ (brighter)
└────┘            └────┘

Medical relevance:
  Different lighting conditions
  Camera flash, natural light, indoor/outdoor
```

**4. Color Jitter** (HSV modification):
```python
# Convert RGB → HSV
hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

# Modify Hue (color shift)
hsv[:,:,0] = hsv[:,:,0] * 0.9  # Slight color change

# Modify Saturation (vividness)
hsv[:,:,1] = hsv[:,:,1] * 1.1  # More saturated

# Modify Value (brightness)
hsv[:,:,2] = hsv[:,:,2] * 1.05  # Slightly brighter

# Convert back HSV → RGB
augmented = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

Medical relevance:
  Different skin tones
  Different camera white balance
```

**5. Zoom/Crop**:
```python
# Random crop
x, y = random_coords()
augmented = image[y:y+480, x:x+480]  # Crop to 480×480
augmented = resize(augmented, (640, 640))  # Resize back

Before:                After:
┌───────────┐          ┌─────┐
│   ●   ●   │    →     │ ●   │ (zoomed on one lesion)
│     ●     │          └─────┘
└───────────┘

Medical relevance:
  Close-up vs far shots
  Single lesion vs multiple lesions
```

**6. Mosaic Augmentation** (YOLO special):
```python
# Combine 4 images into one
┌─────┬─────┐
│Img1 │Img2 │
├─────┼─────┤
│Img3 │Img4 │
└─────┴─────┘

Benefits:
  • See multiple context in one image
  • Learn object relationships
  • Better at detecting small objects
```

**7. Cutout/Mixup**:
```python
# Cutout: Randomly mask regions
┌────────────┐
│   ●   ■■   │  ← Black box hides part of image
│     ● ■■   │
└────────────┘

# Mixup: Blend two images
Image1: 60% weight
Image2: 40% weight
Result: α×Image1 + (1-α)×Image2

Benefits:
  • Model learns to ignore occlusions
  • Better robustness to missing data
```

### YOLOv8 Built-in Augmentations

**Configured in `yolo_acne.yaml`**:

```yaml
# Augmentation parameters
hsv_h: 0.015        # Hue augmentation (±1.5%)
hsv_s: 0.7          # Saturation augmentation (±70%)
hsv_v: 0.4          # Value (brightness) augmentation (±40%)
degrees: 10.0       # Image rotation (±10 degrees)
translate: 0.1      # Image translation (±10%)
scale: 0.5          # Image scale (±50%)
shear: 0.0          # Image shear (disabled)
perspective: 0.0    # Image perspective (disabled)
flipud: 0.0         # Flip upside-down (disabled)
fliplr: 0.5         # Flip left-right (50% chance)
mosaic: 1.0         # Mosaic augmentation (100% of batches)
mixup: 0.1          # Mixup augmentation (10% of batches)
```

**During Training**:
```
Every Batch (16 images):

Original images loaded from disk
         ↓
Apply random augmentations:
  - Random flip (50% chance)
  - Random rotation (±10°)
  - Random brightness (±40%)
  - Random zoom (0.5-1.5×)
  - Mosaic (combine 4 images)
         ↓
Feed to model for training

Next Batch:
  Same images, DIFFERENT augmentations!

Model never sees the exact same image twice
→ Better generalization!
```

---

## 12. GPU Acceleration (M2 Chip) {#gpu-acceleration}

### CPU vs GPU

**CPU** (Central Processing Unit):
```
Architecture:
  • Few cores (8-16 performance cores on M2)
  • Each core is VERY powerful
  • Sequential processing

Task: Multiply 1000×1000 matrices

CPU Approach:
  Core 1: Process row 1
  Core 2: Process row 2
  ...
  Core 8: Process row 8

  Then repeat for remaining rows

  Time: ~100ms
```

**GPU** (Graphics Processing Unit):
```
Architecture:
  • MANY cores (1000s on M2 GPU)
  • Each core is simpler
  • Parallel processing

Task: Same 1000×1000 matrices

GPU Approach:
  Core 1: Process element (0,0)
  Core 2: Process element (0,1)
  Core 3: Process element (0,2)
  ...
  Core 1,000,000: Process element (999,999)

  ALL AT ONCE!

  Time: ~2ms  (50× faster!)
```

**Visualization**:
```
CPU:                         GPU:
Sequential                   Parallel
─────────                   ─────────

Task 1 ████                  Task 1 █
Task 2     ████              Task 2 █
Task 3         ████          Task 3 █
Task 4             ████      Task 4 █

Total: 16 units              Total: 4 units
```

### M2 Chip Architecture

**Apple M2 (Unified Memory Architecture)**:

```
┌───────────────────────────────────────────────┐
│                   M2 Chip                     │
│                                               │
│  ┌──────────────┐        ┌─────────────────┐ │
│  │  CPU Cores   │        │   GPU Cores     │ │
│  │  (8-core)    │        │   (10-core)     │ │
│  │              │        │                 │ │
│  │  4 Perf      │        │   Metal API     │ │
│  │  4 Efficient │        │   3.1 TFLOPS    │ │
│  └──────┬───────┘        └────────┬────────┘ │
│         │                         │          │
│         └──────────┬──────────────┘          │
│                    │                         │
│         ┌──────────▼──────────┐              │
│         │  Unified Memory     │              │
│         │  (16GB Shared)      │              │
│         │                     │              │
│         │  CPU ← → GPU        │              │
│         │  No data copies!    │              │
│         └─────────────────────┘              │
└───────────────────────────────────────────────┘

Key Advantage:
  CPU and GPU share same memory
  → No expensive data transfers
  → Faster than discrete GPUs for some tasks
```

### MPS (Metal Performance Shaders)

**What is MPS?**
- Apple's GPU acceleration framework
- Similar to CUDA (NVIDIA) or ROCm (AMD)
- Integrated with PyTorch

**PyTorch on M2**:

```python
import torch

# Check MPS availability
print(torch.backends.mps.is_available())  # True on M2
print(torch.backends.mps.is_built())      # True

# Device selection
device = 'mps'  # Use M2 GPU

# Create tensor on GPU
x = torch.randn(1000, 1000, device=device)

# All operations now use GPU
y = x @ x.T          # Matrix multiply on GPU
z = torch.relu(y)    # ReLU on GPU
```

**YOLOv8 Automatic Detection**:
```python
from ultralytics import YOLO

model = YOLO('yolov8m.pt')

# Ultralytics auto-detects MPS
# No need to specify device!

model.train(
    data='config/yolo_acne.yaml',
    epochs=100,
    # device='mps'  ← Not needed, auto-detected
)

# Under the hood:
#   1. Load batch of images → MPS memory
#   2. Forward pass → MPS GPU
#   3. Calculate loss → MPS GPU
#   4. Backward pass → MPS GPU
#   5. Update weights → MPS GPU
#   All in unified memory, no copies!
```

### Performance Comparison

**Training Speed**:

```
CPU Only (M2):
  Images/second: ~5
  Epoch time: ~6 minutes
  Total (100 epochs): ~10 hours

MPS GPU (M2):
  Images/second: ~50
  Epoch time: ~40 seconds
  Total (100 epochs): ~1 hour

Speedup: 10× faster with GPU!
```

**Batch Size Impact**:
```
Batch Size 1:
  GPU utilization: 20% (wasteful!)
  Time per epoch: 180s

Batch Size 8:
  GPU utilization: 60%
  Time per epoch: 60s

Batch Size 16:  ← Your configuration
  GPU utilization: 85%
  Time per epoch: 40s  ← Sweet spot!

Batch Size 32:
  GPU utilization: 95%
  Time per epoch: 35s (only slightly faster)
  Memory usage: High (risky!)

Conclusion: Batch 16 is optimal for M2
```

### Optimization Tips

**1. Use Larger Batches** (within memory limits):
```python
# Too small
batch=4  → GPU underutilized

# Sweet spot
batch=16  → GPU ~85% utilized ✓

# Too large
batch=64  → Out of memory ✗
```

**2. Enable Mixed Precision** (FP16):
```python
model.train(
    amp=True  # Automatic Mixed Precision
)

# Uses 16-bit floats instead of 32-bit
# → 2× faster, 2× less memory
# → Negligible accuracy loss
```

**3. Pre-fetch Data**:
```python
model.train(
    workers=4  # 4 CPU threads for data loading
)

# While GPU processes batch N:
#   CPU loads batch N+1
# → No GPU idle time!
```

---

## Summary

You've now completed a comprehensive deep dive into:

1. **Loss Functions**: BCE, IoU, CIoU, and how YOLOv8 combines them
2. **Evaluation Metrics**: Precision, recall, F1, mAP - what they mean and when to use them
3. **Transfer Learning**: How pre-trained models accelerate your training
4. **Data Augmentation**: Creating variations to improve generalization
5. **GPU Acceleration**: M2 architecture and MPS performance benefits

**Combined with Part 1**, you now understand the complete ML pipeline from pixels to predictions!

🎓 **Ready for your presentation on Friday!**
