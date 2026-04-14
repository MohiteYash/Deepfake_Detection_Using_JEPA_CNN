"""
Training script for CNN-JEPA Deepfake Detection

Implements:
- Self-supervised training on real images only
- EMA updates for target encoder
- Loss computation and backpropagation
- Checkpointing and logging
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import numpy as np
import os
import random
from tqdm import tqdm
import argparse

from config import Config
from dataset import create_dataloaders
from model import CNNJEPA
from utils import (
    JEPALoss, 
    save_checkpoint, 
    load_checkpoint, 
    plot_training_curves,
    Logger
)


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_optimizer(model, config):
    """
    Create optimizer
    
    Args:
        model: Model to optimize
        config: Configuration
        
    Returns:
        optimizer: Optimizer instance
    """
    if config.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
    elif config.OPTIMIZER == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
    elif config.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.LEARNING_RATE,
            momentum=0.9,
            weight_decay=config.WEIGHT_DECAY
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.OPTIMIZER}")
    
    return optimizer


def create_scheduler(optimizer, config, num_batches):
    """
    Create learning rate scheduler
    
    Args:
        optimizer: Optimizer instance
        config: Configuration
        num_batches: Number of batches per epoch
        
    Returns:
        scheduler: Scheduler instance or None
    """
    if not config.USE_SCHEDULER:
        return None
    
    if config.SCHEDULER_TYPE == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.NUM_EPOCHS,
            eta_min=config.LEARNING_RATE * 0.01
        )
    elif config.SCHEDULER_TYPE == 'step':
        scheduler = StepLR(
            optimizer,
            step_size=config.NUM_EPOCHS // 3,
            gamma=0.1
        )
    else:
        scheduler = None
    
    return scheduler


def train_epoch(model, dataloader, criterion, optimizer, config, epoch, logger):
    """
    Train for one epoch
    
    Args:
        model: JEPA model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        config: Configuration
        epoch: Current epoch number
        logger: Logger instance
        
    Returns:
        avg_loss_dict: Dictionary of average losses
    """
    model.train()
    
    epoch_losses = {
        'total': [],
        'mse': [],
        'cosine': [],
        'region_consistency': []
    }
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{config.NUM_EPOCHS}')
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        patches = batch['patches'].to(config.DEVICE)
        context_indices = batch['context_indices'].to(config.DEVICE)
        target_indices = batch['target_indices'].to(config.DEVICE)
        
        # Forward pass
        predictions, targets, context_embeddings = model(
            patches, context_indices, target_indices
        )
        
        # Compute loss
        loss, loss_dict = criterion(predictions, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        # Update target encoder with EMA
        model.update_target_encoder(config.EMA_MOMENTUM)
        
        # Record losses
        for key in epoch_losses.keys():
            epoch_losses[key].append(loss_dict[key])
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}",
            'mse': f"{loss_dict['mse']:.4f}",
            'cosine': f"{loss_dict['cosine']:.4f}"
        })
        
        # Log periodically
        if (batch_idx + 1) % config.LOG_INTERVAL == 0:
            msg = (f"Epoch [{epoch}/{config.NUM_EPOCHS}] "
                   f"Batch [{batch_idx + 1}/{len(dataloader)}] "
                   f"Loss: {loss_dict['total']:.4f} "
                   f"MSE: {loss_dict['mse']:.4f} "
                   f"Cosine: {loss_dict['cosine']:.4f} "
                   f"Region: {loss_dict['region_consistency']:.4f}")
            logger.log(msg)
    
    # Compute average losses
    avg_loss_dict = {key: np.mean(values) for key, values in epoch_losses.items()}
    
    return avg_loss_dict


def train(config, resume_from=None):
    """
    Main training function
    
    Args:
        config: Configuration object
        resume_from: Checkpoint path to resume from (optional)
    """
    # Set random seed
    set_seed(config.SEED)
    
    # Create directories
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    # Initialize logger
    logger = Logger(os.path.join(config.LOG_DIR, 'training.log'))
    
    logger.log("=" * 80)
    logger.log("CNN-JEPA Training for Deepfake Detection")
    logger.log("=" * 80)
    
    # Print configuration
    config.print_config()
    
    # Create dataloaders
    logger.log("\nCreating dataloaders...")
    train_loader, eval_loader, patch_sampler = create_dataloaders(config)
    logger.log(f"Training batches: {len(train_loader)}")
    logger.log(f"Evaluation batches: {len(eval_loader)}")
    
    # Create model
    logger.log("\nInitializing model...")
    model = CNNJEPA(config).to(config.DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"Total parameters: {total_params:,}")
    logger.log(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer
    optimizer = create_optimizer(model, config)
    logger.log(f"Optimizer: {config.OPTIMIZER}")
    
    # Create scheduler
    scheduler = create_scheduler(optimizer, config, len(train_loader))
    if scheduler:
        logger.log(f"Scheduler: {config.SCHEDULER_TYPE}")
    
    # Create loss function
    criterion = JEPALoss(
        mse_weight=config.MSE_WEIGHT,
        cosine_weight=config.COSINE_WEIGHT,
        region_weight=config.REGION_CONSISTENCY_WEIGHT
    )
    
    # Resume from checkpoint if specified
    start_epoch = 1
    train_losses = []
    
    if resume_from and os.path.exists(resume_from):
        logger.log(f"\nResuming from checkpoint: {resume_from}")
        start_epoch, train_losses = load_checkpoint(
            model, optimizer, resume_from, config.DEVICE
        )
        start_epoch += 1
    
    # Training loop
    logger.log("\n" + "=" * 80)
    logger.log("Starting training...")
    logger.log("=" * 80 + "\n")
    
    best_loss = float('inf')
    
    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
        # Train one epoch
        avg_loss_dict = train_epoch(
            model, train_loader, criterion, optimizer, config, epoch, logger
        )
        
        # Record losses
        train_losses.append(avg_loss_dict)
        
        # Log epoch summary
        logger.log(f"\nEpoch {epoch} Summary:")
        logger.log(f"  Total Loss: {avg_loss_dict['total']:.6f}")
        logger.log(f"  MSE Loss: {avg_loss_dict['mse']:.6f}")
        logger.log(f"  Cosine Loss: {avg_loss_dict['cosine']:.6f}")
        logger.log(f"  Region Consistency Loss: {avg_loss_dict['region_consistency']:.6f}")
        
        # Update scheduler
        if scheduler:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            logger.log(f"  Learning Rate: {current_lr:.6f}")
        
        # Save checkpoint
        if epoch % config.SAVE_INTERVAL == 0 or epoch == config.NUM_EPOCHS:
            checkpoint_path = os.path.join(
                config.CHECKPOINT_DIR, 
                f'checkpoint_epoch_{epoch}.pth'
            )
            save_checkpoint(model, optimizer, epoch, train_losses, config, checkpoint_path)
        
        # Save best model
        if avg_loss_dict['total'] < best_loss:
            best_loss = avg_loss_dict['total']
            best_model_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, train_losses, config, best_model_path)
            logger.log(f"  New best model saved! Loss: {best_loss:.6f}")
        
        logger.log("")
    
    # Training completed
    logger.log("=" * 80)
    logger.log("Training completed!")
    logger.log("=" * 80)
    
    # Plot training curves
    logger.log("\nGenerating training curves...")
    plot_path = os.path.join(config.RESULTS_DIR, 'training_curves.png')
    plot_training_curves(train_losses, save_path=plot_path)
    logger.log(f"Training curves saved: {plot_path}")
    
    # Save final model
    final_model_path = os.path.join(config.CHECKPOINT_DIR, 'final_model.pth')
    save_checkpoint(model, optimizer, config.NUM_EPOCHS, train_losses, config, final_model_path)
    logger.log(f"Final model saved: {final_model_path}")
    
    return model, train_losses


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train CNN-JEPA for Deepfake Detection')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    
    args = parser.parse_args()
    
    # Override config if specified
    if args.epochs:
        Config.NUM_EPOCHS = args.epochs
    if args.batch_size:
        Config.BATCH_SIZE = args.batch_size
    if args.lr:
        Config.LEARNING_RATE = args.lr
    
    # Train
    train(Config, resume_from=args.resume)


if __name__ == "__main__":
    main()
