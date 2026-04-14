"""
Demo script to test CNN-JEPA implementation with synthetic data

Creates dummy dataset and runs a quick training/evaluation cycle
"""

import os
import numpy as np
from PIL import Image
import torch
from config import Config


def create_synthetic_dataset(num_real=100, num_fake=50):
    """
    Create synthetic dataset for testing
    
    Args:
        num_real: Number of synthetic real images
        num_fake: Number of synthetic fake images
    """
    print("Creating synthetic dataset...")
    
    # Create directories
    os.makedirs('dataset/real', exist_ok=True)
    os.makedirs('dataset/fake', exist_ok=True)
    
    # Generate real images (smooth gradients)
    for i in range(num_real):
        # Create smooth gradient image
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Random gradient direction
        direction = np.random.choice(['horizontal', 'vertical', 'radial'])
        
        if direction == 'horizontal':
            for x in range(224):
                img[:, x, :] = int(255 * x / 224)
        elif direction == 'vertical':
            for y in range(224):
                img[y, :, :] = int(255 * y / 224)
        else:  # radial
            center = (112, 112)
            for y in range(224):
                for x in range(224):
                    dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                    img[y, x, :] = int(255 * min(dist / 112, 1.0))
        
        # Add some noise
        noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Add random color tint
        color_tint = np.random.randint(0, 50, 3)
        img = np.clip(img.astype(np.int16) + color_tint, 0, 255).astype(np.uint8)
        
        # Save
        Image.fromarray(img).save(f'dataset/real/real_{i:04d}.png')
    
    # Generate fake images (with artifacts/discontinuities)
    for i in range(num_fake):
        # Start with similar smooth image
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        
        direction = np.random.choice(['horizontal', 'vertical', 'radial'])
        
        if direction == 'horizontal':
            for x in range(224):
                img[:, x, :] = int(255 * x / 224)
        elif direction == 'vertical':
            for y in range(224):
                img[y, :, :] = int(255 * y / 224)
        else:
            center = (112, 112)
            for y in range(224):
                for x in range(224):
                    dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                    img[y, x, :] = int(255 * min(dist / 112, 1.0))
        
        # Add artifacts (simulating deepfake artifacts)
        # 1. Random blocks with different colors (splicing artifacts)
        num_artifacts = np.random.randint(2, 5)
        for _ in range(num_artifacts):
            x = np.random.randint(0, 168)  # 224 - 56
            y = np.random.randint(0, 168)
            size = 56  # One patch size
            artifact = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
            img[y:y+size, x:x+size] = artifact
        
        # 2. Add sharp discontinuities
        split_line = np.random.randint(56, 168)
        if np.random.rand() > 0.5:
            # Horizontal split
            img[split_line:, :, :] = 255 - img[split_line:, :, :]
        else:
            # Vertical split
            img[:, split_line:, :] = 255 - img[:, split_line:, :]
        
        # Add noise
        noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Save
        Image.fromarray(img).save(f'dataset/fake/fake_{i:04d}.png')
    
    print(f"Created {num_real} real and {num_fake} fake synthetic images")
    print(f"Dataset location: dataset/")


def run_quick_demo():
    """Run a quick demo with minimal epochs"""
    print("\n" + "=" * 80)
    print("RUNNING QUICK DEMO")
    print("=" * 80 + "\n")
    
    # Modify config for quick demo
    Config.NUM_EPOCHS = 5
    Config.BATCH_SIZE = 8
    Config.SAVE_INTERVAL = 5
    
    print("Modified config for quick demo:")
    print(f"  Epochs: {Config.NUM_EPOCHS}")
    print(f"  Batch size: {Config.BATCH_SIZE}")
    
    # Import after modifying config
    from main import run_training, run_evaluation
    
    # Train
    print("\n--- Training Phase ---")
    model = run_training(Config)
    
    # Evaluate
    print("\n--- Evaluation Phase ---")
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth')
    
    if os.path.exists(checkpoint_path):
        results = run_evaluation(Config, checkpoint_path)
        
        print("\n" + "=" * 80)
        print("DEMO RESULTS SUMMARY")
        print("=" * 80)
        print(f"Accuracy:  {results['metrics']['accuracy']:.4f}")
        print(f"Precision: {results['metrics']['precision']:.4f}")
        print(f"Recall:    {results['metrics']['recall']:.4f}")
        print(f"F1 Score:  {results['metrics']['f1']:.4f}")
        print("=" * 80)
    else:
        print(f"\nWarning: Checkpoint not found at {checkpoint_path}")


def test_single_component():
    """Test individual components"""
    print("\n" + "=" * 80)
    print("TESTING INDIVIDUAL COMPONENTS")
    print("=" * 80 + "\n")
    
    # Test dataset loading
    print("1. Testing dataset loading...")
    from dataset import create_dataloaders
    train_loader, eval_loader, patch_sampler = create_dataloaders(Config)
    print(f"   ✓ Train loader: {len(train_loader)} batches")
    print(f"   ✓ Eval loader: {len(eval_loader)} batches")
    
    # Test model
    print("\n2. Testing model...")
    from model import CNNJEPA
    model = CNNJEPA(Config).to(Config.DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Model created with {total_params:,} parameters")
    
    # Test forward pass
    if len(train_loader) > 0:
        print("\n3. Testing forward pass...")
        batch = next(iter(train_loader))
        patches = batch['patches'].to(Config.DEVICE)
        context_indices = batch['context_indices'].to(Config.DEVICE)
        target_indices = batch['target_indices'].to(Config.DEVICE)
        
        predictions, targets, context_embeddings = model(
            patches, context_indices, target_indices
        )
        print(f"   ✓ Predictions shape: {predictions.shape}")
        print(f"   ✓ Targets shape: {targets.shape}")
        
        # Test loss
        print("\n4. Testing loss function...")
        from utils import JEPALoss
        criterion = JEPALoss()
        loss, loss_dict = criterion(predictions, targets)
        print(f"   ✓ Total loss: {loss_dict['total']:.4f}")
        print(f"   ✓ MSE loss: {loss_dict['mse']:.4f}")
        print(f"   ✓ Cosine loss: {loss_dict['cosine']:.4f}")
        
        # Test EMA update
        print("\n5. Testing EMA update...")
        model.update_target_encoder(Config.EMA_MOMENTUM)
        print(f"   ✓ Target encoder updated")
    
    print("\n" + "=" * 80)
    print("ALL COMPONENT TESTS PASSED!")
    print("=" * 80)


def main():
    """Main demo function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Demo script for CNN-JEPA')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['dataset', 'test', 'full'],
                       help='Demo mode: dataset (create only), test (test components), full (create + train + eval)')
    parser.add_argument('--num_real', type=int, default=100,
                       help='Number of synthetic real images')
    parser.add_argument('--num_fake', type=int, default=50,
                       help='Number of synthetic fake images')
    
    args = parser.parse_args()
    
    if args.mode in ['dataset', 'full']:
        create_synthetic_dataset(args.num_real, args.num_fake)
    
    if args.mode in ['test', 'full']:
        test_single_component()
    
    if args.mode == 'full':
        run_quick_demo()
    
    print("\n✅ Demo completed successfully!")
    print("\nNext steps:")
    print("1. Replace synthetic data with real deepfake dataset")
    print("2. Adjust hyperparameters in config.py")
    print("3. Run full training: python main.py --mode train")
    print("4. Evaluate: python main.py --mode eval --checkpoint checkpoints/best_model.pth")


if __name__ == "__main__":
    main()
