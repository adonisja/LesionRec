# Setup Guide for LesionRec - Acne Detection

This guide will walk you through setting up the LesionRec acne detection project from scratch.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [First-Time Setup (Project Owner)](#first-time-setup-project-owner)
3. [Team Member Setup](#team-member-setup)
4. [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Software

- **Python 3.8+**: Check with `python --version`
- **Git**: Check with `git --version`
- **Git LFS**: Install via `brew install git-lfs` (macOS) or `apt-get install git-lfs` (Ubuntu)

### Install Python Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install DVC with Google Drive support
pip install dvc dvc-gdrive

# Install Kaggle API (for dataset downloads)
pip install kaggle
```

## First-Time Setup (Project Owner)

If you're setting up this project for the first time:

### Step 1: Set Up Kaggle API

Most acne datasets are on Kaggle:

```bash
# 1. Go to https://www.kaggle.com/settings/account
# 2. Click "Create New Token" under API section
# 3. This downloads kaggle.json

# 4. Move to ~/.kaggle/
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 5. Test it
kaggle datasets list
```

### Step 2: Initialize Git LFS

```bash
# Git LFS should already be initialized, but verify:
git lfs install
git lfs track "data/samples/*.jpg"
git lfs track "data/samples/*.jpeg"
git lfs track "data/samples/*.png"
```

### Step 3: Set Up DVC

```bash
# Option A: Use the automated script
bash scripts/setup_dvc.sh

# Option B: Manual setup
dvc init
dvc config core.autostage true
dvc config core.analytics false
```

### Step 4: Configure Google Drive Remote

1. **Create a Google Drive folder**:
   - Go to Google Drive
   - Create a folder named "LesionRec_Acne_Data"
   - Open the folder and copy the ID from the URL:
     ```
     https://drive.google.com/drive/folders/1a2b3c4d5e6f7g8h9i0j
                                           ^^^^^^^^^^^^^^^^^^^^
                                           This is your folder ID
     ```

2. **Add the remote to DVC**:
   ```bash
   dvc remote add -d gdrive gdrive://YOUR_FOLDER_ID_HERE
   ```

3. **Update the config file**:
   Edit `.dvc/config` and replace `YOUR_GOOGLE_DRIVE_FOLDER_ID_HERE` with your actual folder ID.

4. **Commit the configuration**:
   ```bash
   git add .dvc/config
   git commit -m "Configure DVC remote storage for acne datasets"
   git push
   ```

### Step 5: Download Acne Datasets

```bash
# Download all acne-specific datasets
python scripts/download_acne_datasets.py --all

# Filter broad datasets for acne only
python scripts/download_acne_datasets.py --filter-acne

# Create unified dataset
python scripts/download_acne_datasets.py --create-unified
```

Note: Some datasets require manual download due to license agreements. The script will provide URLs and instructions.

### Step 6: Create Sample Images

```bash
# Create ~10 sample images for Git LFS
python scripts/download_acne_datasets.py --create-samples
```

This creates a small set of images that team members can use to test code without downloading full datasets.

### Step 7: Track Data with DVC

```bash
# Track the acne datasets
dvc add data/raw/acne_primary
dvc add data/raw/acne_spots
dvc add data/raw/skin_disease

# The above commands create .dvc files
# Commit these to git
git add data/raw/*.dvc data/raw/.gitignore
git commit -m "Add acne datasets to DVC tracking"

# Push data to Google Drive
dvc push

# Push git changes
git push
```

### Step 8: Set Up Roboflow API (For Ensemble Detector)

Your team member is using Roboflow models. Get API key:

```bash
# 1. Go to https://roboflow.com/
# 2. Sign up/login
# 3. Get API key from account settings
# 4. Store securely (DO NOT commit to git)

# Option: Use environment variable
echo "export ROBOFLOW_API_KEY='your_key_here'" >> ~/.bashrc
source ~/.bashrc
```

### Step 9: Set Up Weights & Biases (Optional)

```bash
# Install W&B
pip install wandb

# Login
wandb login

# Update config/default.yaml with your project name
```

## Team Member Setup

If you're joining an existing project:

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd LesionRec
```

### Step 2: Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install roboflow  # For ensemble detector
```

### Step 3: Pull Sample Images

```bash
# Git LFS files are automatically pulled during clone
# If not, run:
git lfs pull
```

### Step 4: Pull Full Datasets (Optional)

```bash
# First time: authenticate with Google Drive
dvc pull

# You'll be prompted to authenticate in your browser
# After that, data will download from Google Drive (~2GB)
```

Note: Full datasets are large. You can start with sample images in `data/samples/` to test code.

### Step 5: Test Ensemble Detector

```bash
# Get Roboflow API key from team
export ROBOFLOW_API_KEY='your_key'

# Test detection on sample image
python src/ensemble_detector.py $ROBOFLOW_API_KEY data/samples/acne_sample_1.jpg
```

### Step 6: Verify Setup

```bash
# Check directory structure
ls -la data/

# Should see:
# - data/samples/ (small images, ~10MB)
# - data/raw/ (after dvc pull, ~2GB)
# - data/processed/ (created during use)
```

## Quick Start Commands

```bash
# Test with sample data only
python -c "
from src.ensemble_detector import AcneEnsembleDetector
detector = AcneEnsembleDetector(api_key='YOUR_KEY')
result = detector.detect('data/samples/acne_sample_1.jpg')
print(f'Acne count: {result.count}')
print(f'Severity: {detector.assess_severity(result)[\"severity\"]}')
"

# Download new acne dataset
python scripts/download_acne_datasets.py --dataset acne_primary

# Filter dataset for acne only
python scripts/download_acne_datasets.py --filter-acne

# Create unified dataset
python scripts/download_acne_datasets.py --create-unified
```

## Troubleshooting

### Issue: DVC Pull Fails

**Solution 1: Re-authenticate**
```bash
rm .dvc/tmp/gdrive-user-credentials.json
dvc pull
```

**Solution 2: Check remote configuration**
```bash
dvc remote list
dvc config core.remote
```

### Issue: Kaggle Download Fails

**Solution:**
```bash
# Check API is set up
ls ~/.kaggle/kaggle.json

# Test connection
kaggle datasets list

# If permission denied:
chmod 600 ~/.kaggle/kaggle.json
```

### Issue: Git LFS Files Not Downloading

**Solution:**
```bash
git lfs pull
git lfs ls-files  # Verify files are tracked
```

### Issue: Permission Denied on Google Drive

**Solution:**
- Ensure the Google Drive folder is shared with your account
- Ask project owner to add you as a collaborator
- Re-authenticate: `rm .dvc/tmp/gdrive-user-credentials.json && dvc pull`

### Issue: Roboflow API Errors

**Solution:**
```bash
# Check API key is set
echo $ROBOFLOW_API_KEY

# Test with simple request
python -c "from roboflow import Roboflow; rf = Roboflow(api_key='YOUR_KEY'); print('✓ API key works')"
```

### Issue: Import Errors

**Solution:**
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

## Directory Structure After Setup

```
LesionRec/
├── .dvc/
│   ├── config              # DVC configuration (has Google Drive ID)
│   └── .gitignore
├── data/
│   ├── raw/
│   │   ├── acne_primary/   # ~200MB (DVC tracked)
│   │   ├── acne_spots/     # ~150MB (DVC tracked)
│   │   └── skin_disease/   # ~50MB filtered (DVC tracked)
│   ├── processed/          # Generated during training
│   │   └── acne_unified/   # Combined dataset
│   └── samples/            # ~10MB (Git LFS tracked)
│       └── acne_sample_*.jpg
├── models/                 # Saved models
├── logs/                   # Training logs
├── src/
│   └── ensemble_detector.py  # Main detection code
├── scripts/
│   ├── setup_dvc.sh
│   └── download_acne_datasets.py
├── config/
│   └── default.yaml        # Configuration
├── .gitignore
├── .gitattributes          # Git LFS configuration
├── requirements.txt
├── README.md
└── ACNE_DETECTION_PIVOT.md  # Strategy document
```

## Next Steps

1. **Explore the data**: Check `data/samples/` for acne images
2. **Test ensemble detector**: Run detection on sample images
3. **Review strategy**: Read [ACNE_DETECTION_PIVOT.md](ACNE_DETECTION_PIVOT.md)
4. **Track experiments**: Set up Weights & Biases
5. **Build API**: Integrate ensemble detector into FastAPI

## Getting Help

- Check [README.md](README.md) for project overview
- Check [data/README.md](data/README.md) for data-specific questions
- Check [QUICK_START_ACNE.md](QUICK_START_ACNE.md) for quick reference
- Open an issue on GitHub
- Ask on team Slack/Discord

## Useful Commands

### DVC Commands

```bash
# Pull all data
dvc pull

# Pull specific dataset
dvc pull data/raw/acne_primary.dvc

# Push your changes to Google Drive
dvc push

# Check status
dvc status

# Update to latest data version
dvc checkout
```

### Git LFS Commands

```bash
# List LFS files
git lfs ls-files

# Pull LFS files
git lfs pull

# Check LFS status
git lfs status
```

### Kaggle Commands

```bash
# List datasets
kaggle datasets list

# Download specific dataset
kaggle datasets download -d nayanchaure/acne-dataset
```

## Project-Specific Notes

### Acne Detection Approach

This project uses an **ensemble of 3 Roboflow models**:
- **acnedet-v1**: Primary detector (best for crisp images)
- **skin_disease_ak**: Classifier/validator
- **skn-1**: Fallback detector for challenging images

See [src/ensemble_detector.py](src/ensemble_detector.py) for implementation.

### Datasets Focus

We focus on **acne-only datasets**:
- ✅ Acne Dataset (Kaggle) - 1,800 images
- ✅ Acne-Wrinkles-Spots - 500 acne images
- ✅ Skin Disease (filtered) - 500 acne images
- ✅ FitzPatrick17k (testing only) - 100 acne images

**Removed datasets**:
- ❌ HAM10000 (melanoma/cancer, not acne)
- ❌ ISIC Archive (skin cancer, not acne)

### Fairness Testing

Critical: Test on FitzPatrick17k to ensure model works across all skin tones (Fitzpatrick I-VI).

```bash
# Download FitzPatrick17k (filtered for acne)
python scripts/download_acne_datasets.py --dataset fitzpatrick --filter-acne

# Test fairness (in development)
# python scripts/test_fairness.py
```
