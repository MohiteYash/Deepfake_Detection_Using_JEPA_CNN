# CNN-JEPA for Deepfake Detection

A research-grade implementation of a **Joint Embedding Predictive Architecture (JEPA)** using CNNs for anomaly-based deepfake detection. The model is trained in a **self-supervised** manner on real images only and detects fake images as deviations from learned representations.

## Novel Contributions

### 1. **Region-Aware JEPA**
- Splits images into grid patches (4x4 = 16 patches)
- Context patches: 60% randomly selected patches
- Target patches: Remaining 40% patches
- Model predicts target patch embeddings from context
- Enables **spatial explainability** via patch-wise anomaly scores

### 2. **Adaptive Thresholding**
- Dynamic threshold computation: `threshold = mean + k * std`
- Based on training data statistics
- More robust than fixed thresholds
- Accounts for dataset-specific variations

### 3. **Region Consistency Loss**
- Novel loss component encouraging consistent predictions across neighboring patches
- Helps model learn coherent spatial representations
- Improves detection of subtle local manipulations

## Architecture

```
Input Image (224×224)
    ↓
Grid Splitting (4×4 patches, 56×56 each)
    ↓
┌─────────────────────────────────────────┐
│ Context Patches (60%)   Target Patches  │
│         ↓                     ↓         │
│   CNN Encoder          Target Encoder   │
│         ↓               (EMA updated)    │
│   Context Encoder            ↓          │
│         ↓              Target Embeddings │
│  Spatial Attention           ↓          │
│         ↓                    ↓          │
│   Predictor Network ────→ Compare       │
│         ↓                    ↓          │
│  Predicted Embeddings    MSE Loss       │
└─────────────────────────────────────────┘
```

### Components

1. **CNN Encoder**: ResNet-18/34 or custom CNN backbone
2. **Context Encoder**: Processes visible patches with optional spatial attention
3. **Target Encoder**: EMA-updated copy of encoder (no gradients)
4. **Predictor Network**: 3-layer MLP predicting target embeddings
5. **Spatial Attention**: Multi-head attention for feature refinement

## Project Structure

```
cnn_jepa_deepfake/
├── config.py              # Configuration and hyperparameters
├── dataset.py             # Dataset loading and patch sampling
├── model.py               # CNN-JEPA architecture
├── utils.py               # Loss functions, metrics, visualization
├── train.py               # Training script
├── evaluate.py            # Evaluation and anomaly detection
├── main.py                # Main runner script
├── requirements.txt       # Python dependencies
├── README.md             # This file
│
├── dataset/              # Dataset directory (create this)
│   ├── real/            # Real images for training
│   └── fake/            # Fake images for evaluation
│
├── checkpoints/          # Saved model checkpoints (auto-created)
├── results/              # Evaluation results and plots (auto-created)
└── logs/                 # Training logs (auto-created)
```

## Quick Start

### 1. Installation

```bash
# Clone or download the project
cd cnn_jepa_deepfake

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

Organize your dataset as follows:

```
dataset/
├── real/
│   ├── real_001.jpg
│   ├── real_002.jpg
│   └── ...
└── fake/
    ├── fake_001.jpg
    ├── fake_002.jpg
    └── ...
```

**Important**: 
- Training uses ONLY `dataset/real/` (self-supervised)
- Evaluation uses both `dataset/real/` and `dataset/fake/`

### 3. Training

```bash
# Train from scratch
python main.py --mode train

# Train with custom parameters
python main.py --mode train --epochs 100 --batch_size 32 --lr 0.0001

# Resume training from checkpoint
python main.py --mode train --resume checkpoints/checkpoint_epoch_20.pth
```

### 4. Evaluation

```bash
# Evaluate trained model
python main.py --mode eval --checkpoint checkpoints/best_model.pth
```

### 5. Inference on Single Image

```bash
# Run inference
python main.py --mode infer --checkpoint checkpoints/best_model.pth --image test.jpg
```

### 6. Full Pipeline

```bash
# Train then evaluate
python main.py --mode full
```

## Configuration

Edit `config.py` to customize:

### Data Settings
- `IMAGE_SIZE`: Input image size (default: 224)
- `PATCH_GRID_SIZE`: Grid size for patches (default: 4, creating 16 patches)
- `CONTEXT_RATIO`: Ratio of context patches (default: 0.6)

### Model Architecture
- `ENCODER_TYPE`: CNN backbone ("resnet18", "resnet34", "custom_cnn")
- `EMBEDDING_DIM`: Embedding dimension (default: 512)
- `USE_ATTENTION`: Enable spatial attention (default: True)

### Training
- `BATCH_SIZE`: Batch size (default: 16)
- `NUM_EPOCHS`: Number of epochs (default: 50)
- `LEARNING_RATE`: Learning rate (default: 1e-4)
- `EMA_MOMENTUM`: EMA momentum for target encoder (default: 0.996)

### Loss Weights
- `MSE_WEIGHT`: MSE loss weight (default: 1.0)
- `COSINE_WEIGHT`: Cosine similarity loss weight (default: 0.5)
- `REGION_CONSISTENCY_WEIGHT`: Region consistency loss weight (default: 0.3)

### Anomaly Detection
- `THRESHOLD_METHOD`: "adaptive" or "fixed"
- `ADAPTIVE_K`: Number of std deviations for adaptive threshold (default: 2.0)

##  Outputs

### During Training
- **Checkpoints**: Saved in `checkpoints/`
  - `checkpoint_epoch_N.pth`: Regular checkpoints
  - `best_model.pth`: Best model (lowest loss)
  - `final_model.pth`: Final model after all epochs
  
- **Logs**: Saved in `logs/training.log`

- **Training Curves**: `results/training_curves.png`
  - Total loss, MSE loss, Cosine loss, Region consistency loss

### During Evaluation
- **Metrics**: Accuracy, Precision, Recall, F1 Score
  
- **Visualizations**:
  - `results/roc_curve.png`: ROC curve with AUC
  - `results/score_distribution.png`: Score distributions for real vs fake
  - `results/anomaly_maps/`: Patch-wise anomaly visualizations
  
- **Results JSON**: `results/evaluation_results.json`
  - Predictions, scores, labels, metrics, threshold

##  Understanding the Method

### Self-Supervised Training (JEPA Phase)

1. **Input**: Real image
2. **Split**: Divide into 4×4 grid (16 patches)
3. **Sample**: Randomly select 60% as context, 40% as target
4. **Encode**: 
   - Context patches → CNN Encoder → Context Encoder
   - Target patches → Target Encoder (EMA)
5. **Predict**: Predictor predicts target embeddings from context
6. **Loss**: MSE + Cosine + Region Consistency
7. **Update**: 
   - Backprop through encoder & predictor
   - EMA update target encoder

### Anomaly Detection (Inference Phase)

1. **Compute Scores**: For each image, compute prediction error
2. **Adaptive Threshold**: 
   - Fit on training (real) data: `threshold = mean + k * std`
3. **Classify**: 
   - Score > threshold → FAKE
   - Score ≤ threshold → REAL
4. **Explain**: Identify patches with highest error

### Why This Works for Deepfakes

- **Real images**: Model learns consistent spatial relationships
- **Fake images**: Inconsistent patches (e.g., face swap boundaries) → high prediction error
- **Region-aware**: Localized manipulations detected via patch scores
- **Self-supervised**: No labels needed during training

## Expected Results

On typical deepfake datasets, you should expect:

- **Training**: Loss steadily decreases, converging around 0.01-0.05
- **Evaluation**: 
  - Accuracy: 85-95%
  - F1 Score: 0.80-0.90
  - AUC: 0.90-0.97

**Note**: Performance depends on:
- Quality and quantity of training data
- Type of deepfakes (GAN-based, face-swap, etc.)
- Hyperparameter tuning

##  Advanced Usage

### Custom CNN Backbone

Edit `model.py` → `CNNEncoder._build_custom_cnn()` to implement your own architecture.

### Custom Loss Functions

Add new loss components in `utils.py` → `JEPALoss.forward()`.

### Different Patch Strategies

Modify `dataset.py` → `PatchSampler` to implement:
- Variable patch sizes
- Overlapping patches
- Hierarchical patching

### Transfer Learning

```python
# Load pretrained weights
checkpoint = torch.load('path/to/checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Fine-tune on new dataset
# ... continue training
```

## Troubleshooting

### Out of Memory
- Reduce `BATCH_SIZE` in config
- Reduce `IMAGE_SIZE` (e.g., 128 instead of 224)
- Use smaller encoder (resnet18 instead of resnet34)

### Poor Performance
- Increase `NUM_EPOCHS` (try 100+)
- Adjust `ADAPTIVE_K` (try 1.5 or 2.5)
- Check dataset balance (should have enough real images)
- Verify image quality

### Slow Training
- Enable CUDA: ensure GPU is available
- Increase `BATCH_SIZE` (if memory allows)
- Reduce `PREDICTOR_DEPTH`

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{cnn_jepa_deepfake,
  title={CNN-JEPA: Region-Aware Self-Supervised Learning for Deepfake Detection},
  author={[Your Name]},
  year={2024},
  note={Research implementation}
}
```

##  Contributing

Contributions welcome! Areas for improvement:
- Additional CNN backbones (EfficientNet, Vision Transformer)
- Multi-scale patch strategies
- Advanced attention mechanisms
- Temporal consistency for video deepfakes

##  License

This code is provided for research purposes. Please cite if used in publications.

## Acknowledgments

- JEPA architecture inspired by [LeCun et al.]
- ResNet backbone from torchvision
- Deepfake detection research community

---

**For questions or issues, please open a GitHub issue or contact [your email].**
