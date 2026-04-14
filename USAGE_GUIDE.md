# CNN-JEPA Deepfake Detection: Complete Usage Guide

## 📋 Table of Contents

1. [Installation](#installation)
2. [Dataset Preparation](#dataset-preparation)
3. [Quick Start](#quick-start)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Inference](#inference)
7. [Understanding Results](#understanding-results)
8. [Hyperparameter Tuning](#hyperparameter-tuning)
9. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM

### Step 1: Install Dependencies

```bash
cd cnn_jepa_deepfake
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Dataset Preparation

### Directory Structure

```
dataset/
├── real/          # Real images (for training)
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
└── fake/          # Fake images (for evaluation only)
    ├── fake001.jpg
    ├── fake002.jpg
    └── ...
```

### Supported Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)

### Recommendations
- **Training set**: 500+ real images minimum, 5000+ recommended
- **Test set**: Balanced mix of real and fake (e.g., 500 real + 500 fake)
- **Image quality**: High resolution (will be resized to 224×224)
- **Diversity**: Various scenes, lighting, subjects

### Using Your Own Dataset

```bash
# Option 1: Copy/move images
cp /path/to/real/images/* dataset/real/
cp /path/to/fake/images/* dataset/fake/

# Option 2: Use symbolic links
ln -s /path/to/real/images dataset/real
ln -s /path/to/fake/images dataset/fake
```

### Demo with Synthetic Data

```bash
# Create synthetic dataset for testing
python demo.py --mode dataset --num_real 100 --num_fake 50
```

---

## Quick Start

### Option 1: Full Demo (Fastest Way to Test)

```bash
# Create synthetic data, test components, train (5 epochs), and evaluate
python demo.py --mode full
```

This will:
1. Create 100 synthetic real images and 50 fake images
2. Test all components
3. Train for 5 epochs (quick demo)
4. Evaluate and show results

### Option 2: Manual Pipeline

```bash
# Step 1: Train
python main.py --mode train --epochs 50

# Step 2: Evaluate
python main.py --mode eval --checkpoint checkpoints/best_model.pth

# Step 3: Test on single image
python main.py --mode infer --checkpoint checkpoints/best_model.pth --image test.jpg
```

### Option 3: All-in-One

```bash
# Train then automatically evaluate
python main.py --mode full
```

---

## Training

### Basic Training

```bash
python main.py --mode train
```

Uses default parameters from `config.py`:
- 50 epochs
- Batch size: 16
- Learning rate: 1e-4
- Adam optimizer

### Custom Training Parameters

```bash
# Train with custom settings
python main.py --mode train \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.0001
```

### Resume Training

```bash
# Resume from checkpoint
python main.py --mode train --resume checkpoints/checkpoint_epoch_20.pth
```

### Training Outputs

```
checkpoints/
├── checkpoint_epoch_5.pth   # Periodic checkpoints
├── checkpoint_epoch_10.pth
├── best_model.pth           # Best model (lowest loss)
└── final_model.pth          # Final model

logs/
└── training.log             # Training log

results/
└── training_curves.png      # Loss curves
```

### Monitoring Training

```bash
# Watch training log in real-time
tail -f logs/training.log
```

### Expected Training Time

| GPU          | Batch Size | Images | Time/Epoch | 50 Epochs |
|--------------|------------|--------|------------|-----------|
| RTX 3090     | 32         | 5000   | ~3 min     | ~2.5 hrs  |
| RTX 2080 Ti  | 16         | 5000   | ~5 min     | ~4 hrs    |
| GTX 1080     | 8          | 5000   | ~10 min    | ~8 hrs    |
| CPU          | 4          | 1000   | ~30 min    | ~25 hrs   |

---

## Evaluation

### Basic Evaluation

```bash
python main.py --mode eval --checkpoint checkpoints/best_model.pth
```

### Evaluation Outputs

```
results/
├── evaluation_results.json    # Detailed results
├── roc_curve.png             # ROC curve
├── score_distribution.png     # Score histograms
└── anomaly_maps/             # Patch-wise visualizations
    ├── sample_0_true_0_pred_0_score_0.0234.png
    ├── sample_1_true_1_pred_1_score_0.1567.png
    └── ...
```

### Understanding Metrics

**evaluation_results.json** contains:

```json
{
  "threshold": 0.0523,
  "metrics": {
    "accuracy": 0.92,
    "precision": 0.89,
    "recall": 0.94,
    "f1": 0.91
  },
  "real_scores": [...],
  "fake_scores": [...]
}
```

- **Threshold**: Automatically computed as `mean + k*std` from training scores
- **Accuracy**: Overall correct predictions
- **Precision**: Of predicted fakes, how many are actually fake
- **Recall**: Of actual fakes, how many are detected
- **F1**: Harmonic mean of precision and recall

### Interpreting ROC Curve

- **AUC = 1.0**: Perfect classifier
- **AUC = 0.9-0.95**: Excellent
- **AUC = 0.8-0.9**: Good
- **AUC = 0.7-0.8**: Fair
- **AUC = 0.5**: Random guessing

---

## Inference

### Single Image Inference

```bash
python main.py --mode infer \
    --checkpoint checkpoints/best_model.pth \
    --image path/to/image.jpg
```

### Output Example

```
================================================================================
RESULTS
--------------------------------------------------------------------------------
Anomaly Score: 0.145632
Threshold: 0.052341
Prediction: FAKE
Confidence: 0.093291

Patch-wise scores:
  Max anomaly: 0.234521 (Patch 7)
  Mean anomaly: 0.145632
  Min anomaly: 0.067234
--------------------------------------------------------------------------------
```

**Interpretation**:
- **Anomaly Score > Threshold** → FAKE
- **High patch score** → That region is suspicious
- **Confidence** = |score - threshold|, higher = more confident

### Batch Inference

Create a custom script:

```python
from main import run_inference
from config import Config
import glob

checkpoint = "checkpoints/best_model.pth"
image_paths = glob.glob("test_images/*.jpg")

for img_path in image_paths:
    score, pred = run_inference(Config, checkpoint, img_path)
    print(f"{img_path}: {pred} (score: {score:.4f})")
```

---

## Understanding Results

### Training Curves

`results/training_curves.png` shows 4 plots:

1. **Total Loss**: Combined loss, should decrease smoothly
2. **MSE Loss**: Reconstruction error, should converge to ~0.01-0.05
3. **Cosine Loss**: Similarity loss, should converge to ~0.1-0.3
4. **Region Consistency**: Spatial coherence, should converge to ~0.2-0.4

**Good training**:
- Smooth decrease
- No oscillations
- Convergence by epoch 40-50

**Bad training**:
- Loss increases or oscillates wildly
- No convergence after 50 epochs
- Loss stuck at high value (>0.1)

### Score Distribution

`results/score_distribution.png`:

**Ideal separation**:
```
Real:  |████████░░░░░░░░░░░|
       0.0      0.05    0.1
       
Fake:           |░░░░░░░░████████|
       0.0      0.05    0.1
```

**Poor separation** (more overlap):
```
Real:  |████████████░░░░░|
Fake:  |░░░░████████████░|
```

### Anomaly Maps

Shows which image patches are most anomalous:

- **Brighter regions**: Higher anomaly score
- **Useful for**: Understanding what the model finds suspicious
- **Example**: In face-swap deepfakes, boundary regions should have high scores

---

## Hyperparameter Tuning

### Key Parameters in `config.py`

#### 1. Architecture

```python
ENCODER_TYPE = "resnet18"  # Try: "resnet34" for better accuracy (slower)
EMBEDDING_DIM = 512        # Try: 256 (faster) or 1024 (better)
USE_ATTENTION = True       # Try: False to disable attention (faster)
```

#### 2. Training

```python
BATCH_SIZE = 16           # Larger = faster but needs more memory
LEARNING_RATE = 1e-4      # Try: 5e-5 (more stable) or 2e-4 (faster)
NUM_EPOCHS = 50           # Try: 100 for better convergence
```

#### 3. Loss Weights

```python
MSE_WEIGHT = 1.0                  # Primary reconstruction loss
COSINE_WEIGHT = 0.5               # Similarity loss
REGION_CONSISTENCY_WEIGHT = 0.3   # Spatial coherence
```

**Tuning tips**:
- Start with defaults
- If overfitting: decrease embedding_dim, increase regularization
- If underfitting: increase epochs, increase embedding_dim

#### 4. Patch Configuration

```python
PATCH_GRID_SIZE = 4       # Try: 8 for finer granularity (slower)
CONTEXT_RATIO = 0.6       # Try: 0.5-0.7
```

#### 5. Anomaly Detection

```python
ADAPTIVE_K = 2.0          # Try: 1.5 (more sensitive) or 2.5 (more conservative)
```

### Tuning for Different Scenarios

**High Precision (fewer false positives)**:
```python
ADAPTIVE_K = 2.5
THRESHOLD_METHOD = "adaptive"
```

**High Recall (catch more fakes)**:
```python
ADAPTIVE_K = 1.5
THRESHOLD_METHOD = "adaptive"
```

**Fast Training**:
```python
BATCH_SIZE = 32
ENCODER_TYPE = "resnet18"
NUM_EPOCHS = 30
USE_ATTENTION = False
```

**Best Accuracy**:
```python
ENCODER_TYPE = "resnet34"
EMBEDDING_DIM = 1024
NUM_EPOCHS = 100
PREDICTOR_DEPTH = 4
```

---

## Troubleshooting

### Issue: Out of Memory Error

**Solution 1**: Reduce batch size
```python
# config.py
BATCH_SIZE = 8  # or even 4
```

**Solution 2**: Reduce image size
```python
# config.py
IMAGE_SIZE = 128  # instead of 224
```

**Solution 3**: Use smaller model
```python
# config.py
ENCODER_TYPE = "resnet18"
EMBEDDING_DIM = 256
```

### Issue: Training Loss Not Decreasing

**Check 1**: Verify dataset
```bash
python demo.py --mode test
```

**Check 2**: Reduce learning rate
```python
LEARNING_RATE = 5e-5  # instead of 1e-4
```

**Check 3**: Increase epochs
```python
NUM_EPOCHS = 100
```

### Issue: Poor Evaluation Performance

**Solution 1**: Check threshold
- Too high threshold → all predicted as real
- Too low threshold → all predicted as fake
- Adjust `ADAPTIVE_K`

**Solution 2**: More training data
- Need 1000+ real images minimum
- Increase diversity

**Solution 3**: Check dataset quality
- Are fake images actually different?
- Is there enough contrast between real/fake?

### Issue: CUDA Out of Memory

```bash
# Reduce batch size
python main.py --mode train --batch_size 4
```

Or use CPU:
```python
# config.py
DEVICE = torch.device("cpu")
```

### Issue: Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

---

## Advanced Tips

### 1. Multi-GPU Training

Modify `train.py`:
```python
model = torch.nn.DataParallel(model)
```

### 2. Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop:
with autocast():
    predictions, targets, _ = model(...)
    loss, _ = criterion(predictions, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3. Custom Dataset Paths

```python
# config.py
DATASET_ROOT = "/path/to/your/dataset"
```

### 4. Early Stopping

Add to `train.py`:
```python
patience = 10
best_loss = float('inf')
counter = 0

for epoch in range(...):
    loss = train_epoch(...)
    
    if loss < best_loss:
        best_loss = loss
        counter = 0
    else:
        counter += 1
    
    if counter >= patience:
        print("Early stopping")
        break
```

---

## Example Workflows

### Workflow 1: Research Experiment

```bash
# 1. Create dataset
# [Place your data in dataset/real and dataset/fake]

# 2. Train
python main.py --mode train --epochs 100

# 3. Evaluate
python main.py --mode eval --checkpoint checkpoints/best_model.pth

# 4. Analyze results
# Check results/evaluation_results.json
# View results/anomaly_maps/
```

### Workflow 2: Production Deployment

```bash
# 1. Train on large dataset
python main.py --mode train --epochs 200 --batch_size 64

# 2. Validate
python main.py --mode eval --checkpoint checkpoints/best_model.pth

# 3. Deploy inference
python main.py --mode infer --checkpoint checkpoints/best_model.pth --image input.jpg
```

### Workflow 3: Hyperparameter Search

```bash
# Try different configurations
for lr in 1e-4 5e-5 2e-4; do
    python main.py --mode train --lr $lr --epochs 50
    python main.py --mode eval --checkpoint checkpoints/best_model.pth
done
```

---

## Contact & Support

For questions, issues, or contributions:
- Open a GitHub issue
- Email: [your-email]
- Documentation: README.md

---

**Happy Research! 🚀**
