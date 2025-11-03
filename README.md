# LesionRec: Acne Detection & Care Recommendation System

A computer vision model for automated acne detection with personalized OTC product recommendations.

## Features

- Acne detection and classification (comedones, papules, pustules, nodules)
- Ensemble ML approach using multiple Roboflow models
- Severity assessment (mild, moderate, severe)
- Personalized OTC care product recommendations
- Skin tone diversity testing for fairness

## Quick Start

### Prerequisites

```bash
# Python 3.8+
python --version

# Install core dependencies
pip install -r requirements.txt

# Install DVC with Google Drive support
pip install dvc dvc-gdrive

# Install Git LFS (if not already installed)
brew install git-lfs  # macOS
# or: apt-get install git-lfs  # Ubuntu
```

### Setup for Team Members

```bash
# 1. Clone the repository
git clone <repo-url>
cd LesionRec

# 2. Install dependencies
pip install -r requirements.txt

# 3. Pull sample data (automatic with git clone)
git lfs pull

# 4. Pull full datasets from Google Drive (first time: authenticate)
dvc pull

# 5. Ready to go!
python src/train.py
```

## Project Structure

```
LesionRec/
├── data/
│   ├── raw/              # Original acne datasets (DVC tracked)
│   ├── processed/        # Preprocessed data (DVC tracked)
│   └── samples/          # Sample images (Git LFS tracked)
├── models/               # Trained models (DVC tracked)
├── notebooks/            # Jupyter notebooks for exploration
├── src/                  # Source code
│   └── ensemble_detector.py  # Ensemble acne detection
├── scripts/              # Utility scripts
│   ├── setup_dvc.sh     # DVC setup automation
│   └── download_acne_datasets.py  # Acne dataset downloader
├── config/               # Configuration files
└── logs/                 # Training logs
```

## Data Management

This project uses a professional ML data workflow:

- **DVC + Google Drive**: Large datasets and models
- **Git LFS**: Small sample images (< 100MB)
- **Download Scripts**: Automated dataset fetching

See [data/README.md](data/README.md) for detailed data management instructions.

### First Time Setup (Project Owner)

```bash
# 1. Set up Kaggle API (for dataset downloads)
mkdir -p ~/.kaggle
# Download kaggle.json from https://www.kaggle.com/settings/account
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 2. Run DVC setup
bash scripts/setup_dvc.sh

# 3. Get your Google Drive folder ID
# Create a folder: "LesionRec_Acne_Data"
# URL: https://drive.google.com/drive/folders/FOLDER_ID

# 4. Configure DVC remote
dvc remote add -d gdrive gdrive://YOUR_FOLDER_ID

# 5. Download acne datasets
python scripts/download_acne_datasets.py --all
python scripts/download_acne_datasets.py --filter-acne
python scripts/download_acne_datasets.py --create-samples

# 6. Track with DVC
dvc add data/raw/acne_primary
dvc add data/raw/acne_spots

# 7. Commit and push
git add .
git commit -m "Setup acne detection data management"
dvc push
git push
```

## Development

### Running Acne Detection

```bash
# Using the ensemble detector
from src.ensemble_detector import AcneEnsembleDetector

# Initialize with your Roboflow API key
detector = AcneEnsembleDetector(api_key="YOUR_ROBOFLOW_API_KEY")

# Detect acne in an image
result = detector.detect("path/to/image.jpg")

print(f"Acne count: {result.count}")
print(f"Confidence: {result.confidence_level}")
print(f"Severity: {detector.assess_severity(result)['severity']}")
```

### Current Model Approach

We use an **ensemble of 3 Roboflow models**:
- **acnedet-v1**: Primary detector (best for crisp head shots)
- **skin_disease_ak**: Classification validator
- **skn-1**: Fallback detector for challenging images

See [ACNE_DETECTION_PIVOT.md](ACNE_DETECTION_PIVOT.md) for detailed strategy.

### Experiment Tracking

We use Weights & Biases for experiment tracking:

```bash
# Install W&B
pip install wandb

# Login (first time)
wandb login

# Track detection experiments
# (Integration with ensemble_detector.py coming soon)
```

### Adding New Data

```bash
# 1. Add data to appropriate directory
cp new_dataset/ data/raw/

# 2. Track with DVC
dvc add data/raw/new_dataset

# 3. Commit
git add data/raw/new_dataset.dvc
git commit -m "Add new dataset"

# 4. Push to Google Drive and Git
dvc push
git push
```

## Datasets

### Primary Datasets

#### Acne Dataset (Kaggle)
- **Size**: ~200MB
- **Images**: ~1,800 acne images
- **Classes**: Various acne types (comedones, papules, pustules)
- **Use**: Primary training dataset
- **Source**: [Kaggle](https://www.kaggle.com/datasets/nayanchaure/acne-dataset)

#### Acne-Wrinkles-Spots Classification
- **Size**: ~150MB
- **Images**: ~500 acne-labeled images
- **Classes**: Multi-label (acne, spots, wrinkles)
- **Use**: Supplementary training data
- **Source**: [Kaggle](https://www.kaggle.com/datasets/ranvijaybalbir/acne-wrinkles-spots-classification)

### Supplementary Datasets

#### Skin Disease Dataset (Filtered)
- **Size**: ~500MB (full), ~50MB (acne-filtered)
- **Images**: ~5,000 total (filter for acne/rosacea)
- **Use**: Additional training data after filtering
- **Source**: [Kaggle](https://www.kaggle.com/datasets/pacificrm/skindiseasedataset)

#### FitzPatrick17k
- **Size**: ~2GB
- **Images**: ~100 acne cases (filtered)
- **Use**: **Diversity/bias testing only** (not for training)
- **Purpose**: Ensure model works across all skin tones
- **Source**: [GitHub](https://github.com/mattgroh/fitzpatrick17k)

## Model Architecture

### Ensemble Detection Strategy

We use a **smart ensemble approach** combining three specialized Roboflow models:

```
┌─────────────────────────────────────────────────┐
│            Input: Acne Image                    │
└────────────────┬────────────────────────────────┘
                 │
                 ├──► Image Quality Assessment
                 │
    ┌────────────┴────────────┐
    │  Preprocessing          │
    │  (CLAHE, Denoise, etc)  │
    └────────────┬────────────┘
                 │
    ┌────────────┴────────────┐
    │                         │
    │   Parallel Detection    │
    │                         │
    ├─► acnedet-v1           │  Primary detector
    ├─► skin_disease_ak      │  Classifier/validator
    └─► skn-1                │  Fallback detector
         │
         ▼
    ┌────────────────────────┐
    │  Ensemble Fusion Logic │
    │  - Agreement scoring   │
    │  - Confidence weighting│
    │  - Quality-based choice│
    └────────────┬───────────┘
                 │
                 ▼
    ┌────────────────────────┐
    │  Detection Result      │
    │  + Confidence Level    │
    │  + Severity Assessment │
    └────────────────────────┘
```

**Key Components**:
- **Image Quality Assessor**: Determines optimal detection strategy
- **Smart Preprocessing**: Enhances low-quality images
- **Ensemble Logic**: Fuses model outputs with confidence weighting
- **Severity Classifier**: Categorizes acne as mild, moderate, or severe

See [src/ensemble_detector.py](src/ensemble_detector.py) for implementation.

## Results

Performance metrics and fairness testing results will be documented here as the project progresses.

**Target Metrics**:
- Precision: > 85%
- Recall: > 80%
- F1 Score: > 82%
- Accuracy variance across skin tones: < 10%

## Contributing

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make changes and commit: `git commit -m "Add feature"`
3. Track new data with DVC if needed: `dvc add data/...`
4. Push changes: `dvc push && git push`
5. Create a Pull Request

## Troubleshooting

### DVC Issues

```bash
# Can't pull data?
dvc doctor  # Check configuration

# Re-authenticate with Google Drive
rm .dvc/tmp/gdrive-user-credentials.json
dvc pull
```

### Git LFS Issues

```bash
# LFS files not downloaded?
git lfs pull

# Check LFS status
git lfs ls-files
```

See [data/README.md](data/README.md) for more troubleshooting tips.

## Resources

### Project Documentation
- [ACNE_DETECTION_PIVOT.md](ACNE_DETECTION_PIVOT.md) - Detailed acne detection strategy
- [QUICK_START_ACNE.md](QUICK_START_ACNE.md) - Quick start guide for team
- [data/README.md](data/README.md) - Data management guide
- [SETUP.md](SETUP.md) - Complete setup instructions

### Tools & Frameworks
- [DVC Documentation](https://dvc.org/doc)
- [Roboflow](https://roboflow.com/) - Model hosting and deployment
- [Weights & Biases](https://wandb.ai/) - Experiment tracking
- [Git LFS](https://git-lfs.github.com/)

### Research Papers
- [Deep Learning for Acne Detection](https://arxiv.org/search/?query=acne+detection&searchtype=all)
- [FitzPatrick17k Paper](https://arxiv.org/abs/2104.09957) - Skin tone diversity

## License

(Add your license here)

## Team

(Add team members here)

## Acknowledgments

- Acne Dataset creators (Kaggle community)
- FitzPatrick17k dataset team
- Roboflow community
- Open source ML community
