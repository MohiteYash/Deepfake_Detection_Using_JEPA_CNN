# 🚀 CNN-JEPA Deepfake Detection - QUICKSTART

## ⚡ Fastest Way to Get Started (5 minutes)

### Step 1: Setup (1 minute)
```bash
cd cnn_jepa_deepfake
pip install -r requirements.txt
```

### Step 2: Run Demo (4 minutes)
```bash
# Creates synthetic data, trains for 5 epochs, evaluates
python demo.py --mode full
```

**That's it!** You now have a trained model and evaluation results.

---

## 📁 What You Get

After running the demo, you'll have:

```
cnn_jepa_deepfake/
├── checkpoints/
│   ├── best_model.pth          ← Trained model
│   └── final_model.pth
│
├── results/
│   ├── training_curves.png      ← Loss curves
│   ├── roc_curve.png            ← ROC curve
│   ├── score_distribution.png   ← Score histograms
│   └── anomaly_maps/            ← Explainability visualizations
│
├── logs/
│   ├── training.log             ← Training details
│   └── evaluation.log           ← Evaluation details
│
└── dataset/
    ├── real/                    ← 100 synthetic real images
    └── fake/                    ← 50 synthetic fake images
```

---

## 🎯 Next: Use Your Own Data

### Step 1: Prepare Your Dataset
```bash
# Create directories
mkdir -p dataset/real dataset/fake

# Add your images
cp /path/to/real/images/* dataset/real/
cp /path/to/fake/images/* dataset/fake/
```

### Step 2: Train
```bash
python main.py --mode train --epochs 50
```

### Step 3: Evaluate
```bash
python main.py --mode eval --checkpoint checkpoints/best_model.pth
```

### Step 4: Test Single Image
```bash
python main.py --mode infer --checkpoint checkpoints/best_model.pth --image test.jpg
```

---

## 📊 Expected Output

### Training
```
Epoch 1/50: Loss: 0.0523 MSE: 0.0412 Cosine: 0.0089 Region: 0.0022
Epoch 2/50: Loss: 0.0389 MSE: 0.0301 Cosine: 0.0067 Region: 0.0021
...
Checkpoint saved: checkpoints/best_model.pth
```

### Evaluation
```
Threshold: 0.0523
Accuracy:  0.92
Precision: 0.89
Recall:    0.94
F1 Score:  0.91
ROC AUC:   0.95
```

### Inference
```
Anomaly Score: 0.145632
Threshold: 0.052341
Prediction: FAKE
Confidence: 0.093291
```

---

## 📚 Full Documentation

- **README.md** - Complete overview and architecture
- **USAGE_GUIDE.md** - Detailed usage instructions
- **PROJECT_OVERVIEW.md** - Technical details and research info

---

## 🔧 Common Commands

```bash
# Train with custom settings
python main.py --mode train --epochs 100 --batch_size 32 --lr 0.0001

# Resume training
python main.py --mode train --resume checkpoints/checkpoint_epoch_20.pth

# Evaluate with visualization
python main.py --mode eval --checkpoint checkpoints/best_model.pth

# Full pipeline (train + eval)
python main.py --mode full

# Create synthetic dataset only
python demo.py --mode dataset --num_real 200 --num_fake 100

# Test components
python demo.py --mode test
```

---

## ⚙️ Key Configuration (config.py)

```python
# Quick tuning options
NUM_EPOCHS = 50              # Training epochs
BATCH_SIZE = 16              # Batch size
LEARNING_RATE = 1e-4         # Learning rate
ADAPTIVE_K = 2.0             # Threshold sensitivity
ENCODER_TYPE = "resnet18"    # Backbone (resnet18/resnet34/custom_cnn)
```

---

## 🆘 Troubleshooting

### Out of memory?
```python
# In config.py
BATCH_SIZE = 4
IMAGE_SIZE = 128
```

### Training too slow?
```bash
# Use GPU if available
python -c "import torch; print(torch.cuda.is_available())"

# Or reduce epochs for quick test
python main.py --mode train --epochs 10
```

### Poor results?
```python
# In config.py
NUM_EPOCHS = 100        # More training
ADAPTIVE_K = 1.5        # More sensitive
```

---

## 📞 Need Help?

1. Check **USAGE_GUIDE.md** for detailed instructions
2. Run `python demo.py --mode test` to verify setup
3. Check logs in `logs/` directory
4. Open GitHub issue with error details

---

## ✨ Features at a Glance

✅ Self-supervised learning (trains on real images only)  
✅ Region-aware prediction (4×4 patch grid)  
✅ Adaptive thresholding (automatic calibration)  
✅ Explainability (patch-wise anomaly maps)  
✅ Complete evaluation (Accuracy, F1, AUC, ROC)  
✅ GPU acceleration (CUDA support)  
✅ Checkpoint management (save/resume)  
✅ Research-grade code (~2,500 lines, well-documented)  

---

**Ready to detect deepfakes? Run `python demo.py --mode full` now! 🚀**
