"""
Utility functions for CNN-JEPA Deepfake Detection

Includes:
- Loss functions
- Evaluation metrics
- Visualization
- Checkpoint management
- Anomaly detection logic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import os
from PIL import Image
import json


# ==================== LOSS FUNCTIONS ====================

class JEPALoss(nn.Module):
    """
    Combined loss for JEPA training
    Includes MSE, Cosine Similarity, and Region Consistency losses
    """
    
    def __init__(self, mse_weight=1.0, cosine_weight=0.5, region_weight=0.3):
        """
        Args:
            mse_weight: Weight for MSE loss
            cosine_weight: Weight for cosine similarity loss
            region_weight: Weight for region consistency loss
        """
        super(JEPALoss, self).__init__()
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight
        self.region_weight = region_weight
        
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineSimilarity(dim=-1)
        
    def forward(self, predictions, targets):
        """
        Compute combined loss
        
        Args:
            predictions: Predicted embeddings [B, N_target, D]
            targets: Target embeddings [B, N_target, D]
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        # MSE Loss
        mse = self.mse_loss(predictions, targets)
        
        # Cosine Similarity Loss (1 - cosine similarity)
        cosine_sim = self.cosine_loss(predictions, targets)  # [B, N_target]
        cosine = (1.0 - cosine_sim).mean()
        
        # Region Consistency Loss
        # Encourage consistent predictions across neighboring patches
        region_consistency = self._compute_region_consistency(predictions)
        
        # Total loss
        total_loss = (self.mse_weight * mse + 
                     self.cosine_weight * cosine + 
                     self.region_weight * region_consistency)
        
        loss_dict = {
            'total': total_loss.item(),
            'mse': mse.item(),
            'cosine': cosine.item(),
            'region_consistency': region_consistency.item()
        }
        
        return total_loss, loss_dict
    
    def _compute_region_consistency(self, predictions):
        """
        Compute consistency between neighboring patch predictions
        
        Args:
            predictions: [B, N_target, D]
            
        Returns:
            consistency_loss: Scalar loss
        """
        B, N, D = predictions.shape
        
        if N < 2:
            return torch.tensor(0.0, device=predictions.device)
        
        # Compute pairwise distances between predictions
        pred_norm = F.normalize(predictions, p=2, dim=-1)
        similarity_matrix = torch.matmul(pred_norm, pred_norm.transpose(1, 2))  # [B, N, N]
        
        # Encourage high similarity (consistency)
        consistency_loss = (1.0 - similarity_matrix).mean()
        
        return consistency_loss


# ==================== EVALUATION METRICS ====================

def compute_metrics(predictions, labels):
    """
    Compute classification metrics
    
    Args:
        predictions: Binary predictions (0/1)
        labels: Ground truth labels (0/1)
        
    Returns:
        metrics: Dictionary of metrics
    """
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, zero_division=0),
        'recall': recall_score(labels, predictions, zero_division=0),
        'f1': f1_score(labels, predictions, zero_division=0)
    }
    
    return metrics


def compute_anomaly_scores(predictions, targets):
    """
    Compute anomaly scores from prediction errors
    
    Args:
        predictions: Predicted embeddings [B, N_target, D]
        targets: Target embeddings [B, N_target, D]
        
    Returns:
        scores: Anomaly scores [B]
    """
    # Compute MSE per sample
    mse_per_sample = ((predictions - targets) ** 2).mean(dim=[1, 2])  # [B]
    
    return mse_per_sample.cpu().numpy()


def compute_patch_anomaly_scores(predictions, targets):
    """
    Compute per-patch anomaly scores for explainability
    
    Args:
        predictions: [B, N_target, D]
        targets: [B, N_target, D]
        
    Returns:
        patch_scores: [B, N_target]
    """
    # MSE per patch
    patch_scores = ((predictions - targets) ** 2).mean(dim=-1)  # [B, N_target]
    
    return patch_scores.cpu().numpy()


# ==================== ADAPTIVE THRESHOLDING ====================

class AdaptiveThreshold:
    """
    Adaptive threshold computation based on training error statistics
    """
    
    def __init__(self, k=2.0):
        """
        Args:
            k: Number of standard deviations above mean
        """
        self.k = k
        self.mean = None
        self.std = None
        self.threshold = None
        
    def fit(self, training_scores):
        """
        Fit threshold based on training scores
        
        Args:
            training_scores: Array of anomaly scores from training data
        """
        training_scores = np.array(training_scores)
        self.mean = np.mean(training_scores)
        self.std = np.std(training_scores)
        self.threshold = self.mean + self.k * self.std
        
        print(f"\nAdaptive Threshold Computed:")
        print(f"  Mean: {self.mean:.6f}")
        print(f"  Std: {self.std:.6f}")
        print(f"  Threshold (mean + {self.k} * std): {self.threshold:.6f}")
        
        return self.threshold
    
    def predict(self, scores):
        """
        Predict anomalies based on threshold
        
        Args:
            scores: Anomaly scores
            
        Returns:
            predictions: Binary predictions (0=real, 1=fake)
        """
        if self.threshold is None:
            raise ValueError("Threshold not fitted. Call fit() first.")
        
        predictions = (np.array(scores) > self.threshold).astype(int)
        return predictions


# ==================== VISUALIZATION ====================

def visualize_anomaly_map(image, patch_scores, patch_positions, save_path=None):
    """
    Visualize patch-wise anomaly map
    
    Args:
        image: Original image tensor [C, H, W]
        patch_scores: Anomaly scores per patch [N_target]
        patch_positions: Positions of target patches
        save_path: Path to save visualization
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean
    image = torch.clamp(image, 0, 1)
    image_np = image.permute(1, 2, 0).cpu().numpy()
    
    # Show original image
    axes[0].imshow(image_np)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Create anomaly heatmap
    H, W = image_np.shape[:2]
    heatmap = np.zeros((H, W))
    
    # This is simplified - in practice, map patch_positions to grid locations
    # For visualization purposes
    axes[1].imshow(image_np)
    axes[1].set_title(f"Anomaly Map (Max Score: {patch_scores.max():.4f})")
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()


def plot_training_curves(train_losses, save_path=None):
    """
    Plot training loss curves
    
    Args:
        train_losses: List of loss dictionaries
        save_path: Path to save plot
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Total loss
    total_losses = [loss['total'] for loss in train_losses]
    axes[0, 0].plot(epochs, total_losses, 'b-', linewidth=2)
    axes[0, 0].set_title('Total Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # MSE loss
    mse_losses = [loss['mse'] for loss in train_losses]
    axes[0, 1].plot(epochs, mse_losses, 'r-', linewidth=2)
    axes[0, 1].set_title('MSE Loss', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Cosine loss
    cosine_losses = [loss['cosine'] for loss in train_losses]
    axes[1, 0].plot(epochs, cosine_losses, 'g-', linewidth=2)
    axes[1, 0].set_title('Cosine Loss', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Region consistency loss
    region_losses = [loss['region_consistency'] for loss in train_losses]
    axes[1, 1].plot(epochs, region_losses, 'm-', linewidth=2)
    axes[1, 1].set_title('Region Consistency Loss', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()


def plot_roc_curve(labels, scores, save_path=None):
    """
    Plot ROC curve
    
    Args:
        labels: True labels
        scores: Anomaly scores
        save_path: Path to save plot
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()
    
    return auc


def plot_score_distribution(real_scores, fake_scores, threshold, save_path=None):
    """
    Plot distribution of anomaly scores for real vs fake
    
    Args:
        real_scores: Scores for real images
        fake_scores: Scores for fake images
        threshold: Decision threshold
        save_path: Path to save plot
    """
    plt.figure(figsize=(10, 6))
    
    plt.hist(real_scores, bins=50, alpha=0.6, label='Real Images', color='blue', density=True)
    plt.hist(fake_scores, bins=50, alpha=0.6, label='Fake Images', color='red', density=True)
    plt.axvline(threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold = {threshold:.4f}')
    
    plt.xlabel('Anomaly Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Distribution of Anomaly Scores', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()


# ==================== CHECKPOINT MANAGEMENT ====================

def save_checkpoint(model, optimizer, epoch, train_losses, config, filename):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        train_losses: Training loss history
        config: Configuration
        filename: Checkpoint filename
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'config': vars(config)
    }
    
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")


def load_checkpoint(model, optimizer, filename, device):
    """
    Load model checkpoint
    
    Args:
        model: Model to load into
        optimizer: Optimizer to load into
        filename: Checkpoint filename
        device: Device to load to
        
    Returns:
        epoch: Epoch number
        train_losses: Training loss history
    """
    checkpoint = torch.load(filename, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    train_losses = checkpoint['train_losses']
    
    print(f"Checkpoint loaded: {filename}")
    print(f"Resuming from epoch {epoch}")
    
    return epoch, train_losses


def save_results(results, filename):
    """
    Save evaluation results to JSON
    
    Args:
        results: Dictionary of results
        filename: Output filename
    """
    # Convert numpy arrays to lists
    results_serializable = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            results_serializable[key] = value.tolist()
        else:
            results_serializable[key] = value
    
    with open(filename, 'w') as f:
        json.dump(results_serializable, f, indent=4)
    
    print(f"Results saved: {filename}")


# ==================== LOGGING ====================

class Logger:
    """Simple logger for training"""
    
    def __init__(self, log_file):
        self.log_file = log_file
        
    def log(self, message):
        """Log message to file and print"""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test loss function
    predictions = torch.randn(4, 6, 512)
    targets = torch.randn(4, 6, 512)
    
    loss_fn = JEPALoss()
    total_loss, loss_dict = loss_fn(predictions, targets)
    print(f"\nLoss test:")
    print(f"  Total: {loss_dict['total']:.4f}")
    print(f"  MSE: {loss_dict['mse']:.4f}")
    print(f"  Cosine: {loss_dict['cosine']:.4f}")
    print(f"  Region: {loss_dict['region_consistency']:.4f}")
    
    # Test adaptive threshold
    training_scores = np.random.randn(100) * 0.1 + 0.5
    threshold_computer = AdaptiveThreshold(k=2.0)
    threshold = threshold_computer.fit(training_scores)
    
    test_scores = np.random.randn(20) * 0.2 + 0.6
    predictions = threshold_computer.predict(test_scores)
    print(f"\nThreshold predictions: {predictions}")
