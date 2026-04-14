"""
Configuration file for CNN-JEPA Deepfake Detection
Contains all hyperparameters and settings
"""

import torch

class Config:
    """Configuration class for CNN-JEPA model"""
    
    # ==================== DATA SETTINGS ====================
    # Dataset paths
    DATASET_ROOT = "dataset"
    REAL_DIR = "dataset/real"
    FAKE_DIR = "dataset/fake"
    
    # Image settings
    IMAGE_SIZE = 224  # Input image size
    NUM_PATCHES = 16  # Grid size for patches (4x4 = 16 patches)
    PATCH_GRID_SIZE = 4  # 4x4 grid
    
    # Patch sampling strategy
    CONTEXT_RATIO = 0.6  # 60% patches as context
    TARGET_RATIO = 0.4   # 40% patches as target
    
    # ==================== MODEL ARCHITECTURE ====================
    # Encoder settings
    ENCODER_TYPE = "resnet18"  # Options: "resnet18", "resnet34", "custom_cnn"
    EMBEDDING_DIM = 512  # Dimension of feature embeddings
    
    # Context encoder
    CONTEXT_HIDDEN_DIM = 1024
    
    # Predictor network
    PREDICTOR_HIDDEN_DIM = 1024
    PREDICTOR_DEPTH = 3  # Number of layers in predictor MLP
    
    # Target encoder (EMA)
    EMA_MOMENTUM = 0.996  # Momentum coefficient for EMA update
    
    # Attention module
    USE_ATTENTION = True
    ATTENTION_HEADS = 8
    
    # ==================== TRAINING SETTINGS ====================
    # Training hyperparameters
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # Loss weights
    MSE_WEIGHT = 1.0
    COSINE_WEIGHT = 0.5
    REGION_CONSISTENCY_WEIGHT = 0.3
    
    # Optimizer
    OPTIMIZER = "adamw"  # Options: "adam", "adamw", "sgd"
    
    # Scheduler
    USE_SCHEDULER = True
    SCHEDULER_TYPE = "cosine"  # Options: "cosine", "step", "none"
    WARMUP_EPOCHS = 5
    
    # ==================== ANOMALY DETECTION ====================
    # Adaptive thresholding
    THRESHOLD_METHOD = "adaptive"  # Options: "adaptive", "fixed"
    FIXED_THRESHOLD = 0.1
    ADAPTIVE_K = 2.0  # threshold = mean + k * std
    
    # Region inconsistency scoring
    REGION_THRESHOLD_PERCENTILE = 90  # Top 10% most anomalous regions
    
    # ==================== EVALUATION SETTINGS ====================
    # Metrics
    SAVE_ANOMALY_MAPS = True
    SAVE_PREDICTIONS = True
    
    # ==================== SYSTEM SETTINGS ====================
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Random seed for reproducibility
    SEED = 42
    
    # Logging
    LOG_INTERVAL = 10  # Log every N batches
    SAVE_INTERVAL = 5  # Save checkpoint every N epochs
    
    # Paths
    CHECKPOINT_DIR = "checkpoints"
    RESULTS_DIR = "results"
    LOG_DIR = "logs"
    
    # ==================== VISUALIZATION ====================
    VIS_NUM_SAMPLES = 10  # Number of samples to visualize
    VIS_SAVE_FORMAT = "png"
    
    @staticmethod
    def print_config():
        """Print all configuration settings"""
        print("=" * 60)
        print("CNN-JEPA Configuration")
        print("=" * 60)
        for attr in dir(Config):
            if not attr.startswith("_") and attr.isupper():
                print(f"{attr}: {getattr(Config, attr)}")
        print("=" * 60)
