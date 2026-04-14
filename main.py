"""
Main runner script for CNN-JEPA Deepfake Detection

Provides end-to-end pipeline:
1. Training
2. Evaluation
3. Inference on new images
"""

import torch
import argparse
import os
import sys

from config import Config
from train import train
from evaluate import evaluate, AnomalyDetector
from dataset import create_dataloaders, get_transforms, PatchSampler
from model import CNNJEPA
from PIL import Image
import numpy as np


def run_training(config, resume_from=None):
    """
    Run training pipeline
    
    Args:
        config: Configuration object
        resume_from: Checkpoint to resume from
    """
    print("\n" + "=" * 80)
    print("TRAINING PIPELINE")
    print("=" * 80 + "\n")
    
    model, train_losses = train(config, resume_from=resume_from)
    
    print("\nTraining completed successfully!")
    print(f"Best model saved to: {os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')}")
    
    return model


def run_evaluation(config, checkpoint_path):
    """
    Run evaluation pipeline
    
    Args:
        config: Configuration object
        checkpoint_path: Path to trained model
    """
    print("\n" + "=" * 80)
    print("EVALUATION PIPELINE")
    print("=" * 80 + "\n")
    
    results = evaluate(config, checkpoint_path, visualize_samples=True)
    
    print("\nEvaluation completed successfully!")
    print(f"Results saved to: {config.RESULTS_DIR}")
    
    return results


def run_inference(config, checkpoint_path, image_path):
    """
    Run inference on a single image
    
    Args:
        config: Configuration object
        checkpoint_path: Path to trained model
        image_path: Path to input image
    """
    print("\n" + "=" * 80)
    print("INFERENCE ON SINGLE IMAGE")
    print("=" * 80 + "\n")
    
    print(f"Image: {image_path}")
    
    # Load model
    print("Loading model...")
    model = CNNJEPA(config).to(config.DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully")
    
    # Load and preprocess image
    print("Preprocessing image...")
    transform = get_transforms('eval')
    image_pil = Image.open(image_path).convert('RGB')
    image = transform(image_pil)
    
    # Create patch sampler
    patch_sampler = PatchSampler(
        image_size=config.IMAGE_SIZE,
        patch_grid_size=config.PATCH_GRID_SIZE,
        context_ratio=config.CONTEXT_RATIO
    )
    
    # Extract patches
    patches, positions = patch_sampler.extract_patches(image)
    
    # Sample context and target
    context_indices, target_indices = patch_sampler.sample_context_target(len(patches))
    
    # Create detector
    detector = AnomalyDetector(model, config)
    
    # Compute anomaly score
    print("Computing anomaly score...")
    anomaly_score, patch_scores, most_anomalous_patch = detector.explain(
        image,
        patches,
        torch.tensor(context_indices),
        torch.tensor(target_indices)
    )
    
    # We need to fit threshold first (using dummy data for demo)
    # In practice, this should be fitted on real training data
    print("\nNote: For proper inference, threshold should be fitted on training data")
    print("      Using default threshold of 0.1 for demonstration")
    
    threshold = 0.1  # Default threshold
    prediction = "FAKE" if anomaly_score > threshold else "REAL"
    
    print("\n" + "-" * 80)
    print("RESULTS")
    print("-" * 80)
    print(f"Anomaly Score: {anomaly_score:.6f}")
    print(f"Threshold: {threshold:.6f}")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {abs(anomaly_score - threshold):.6f}")
    print(f"\nPatch-wise scores:")
    print(f"  Max anomaly: {patch_scores.max():.6f} (Patch {most_anomalous_patch})")
    print(f"  Mean anomaly: {patch_scores.mean():.6f}")
    print(f"  Min anomaly: {patch_scores.min():.6f}")
    print("-" * 80)
    
    return anomaly_score, prediction


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='CNN-JEPA for Deepfake Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train from scratch
  python main.py --mode train
  
  # Resume training
  python main.py --mode train --resume checkpoints/checkpoint_epoch_10.pth
  
  # Evaluate trained model
  python main.py --mode eval --checkpoint checkpoints/best_model.pth
  
  # Run inference on single image
  python main.py --mode infer --checkpoint checkpoints/best_model.pth --image test.jpg
  
  # Full pipeline: train then evaluate
  python main.py --mode full
        """
    )
    
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'eval', 'infer', 'full'],
                       help='Execution mode')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (for eval/infer)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume training from checkpoint')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to image for inference')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Override config if specified
    if args.epochs:
        Config.NUM_EPOCHS = args.epochs
    if args.batch_size:
        Config.BATCH_SIZE = args.batch_size
    if args.lr:
        Config.LEARNING_RATE = args.lr
    
    # Execute based on mode
    if args.mode == 'train':
        run_training(Config, resume_from=args.resume)
        
    elif args.mode == 'eval':
        if not args.checkpoint:
            print("Error: --checkpoint required for evaluation")
            sys.exit(1)
        run_evaluation(Config, args.checkpoint)
        
    elif args.mode == 'infer':
        if not args.checkpoint:
            print("Error: --checkpoint required for inference")
            sys.exit(1)
        if not args.image:
            print("Error: --image required for inference")
            sys.exit(1)
        run_inference(Config, args.checkpoint, args.image)
        
    elif args.mode == 'full':
        # Train
        model = run_training(Config, resume_from=args.resume)
        
        # Evaluate
        best_checkpoint = os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth')
        if os.path.exists(best_checkpoint):
            run_evaluation(Config, best_checkpoint)
        else:
            print("\nWarning: Best model checkpoint not found. Skipping evaluation.")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
