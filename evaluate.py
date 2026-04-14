"""
Evaluation script for CNN-JEPA Deepfake Detection

Implements:
- Anomaly score computation
- Adaptive thresholding
- Classification metrics
- Explainability (patch-wise anomaly maps)
- Visualization
"""

import torch
import numpy as np
import os
from tqdm import tqdm
import argparse
import json

from config import Config
from dataset import create_dataloaders
from model import CNNJEPA
from utils import (
    compute_anomaly_scores,
    compute_patch_anomaly_scores,
    compute_metrics,
    AdaptiveThreshold,
    plot_roc_curve,
    plot_score_distribution,
    visualize_anomaly_map,
    save_results,
    Logger
)


class AnomalyDetector:
    """
    Anomaly detector using trained JEPA model
    """
    
    def __init__(self, model, config):
        """
        Args:
            model: Trained JEPA model
            config: Configuration
        """
        self.model = model
        self.config = config
        self.threshold_computer = AdaptiveThreshold(k=config.ADAPTIVE_K)
        self.threshold = None
        
    def compute_scores(self, dataloader, desc='Computing scores'):
        """
        Compute anomaly scores for all samples
        
        Args:
            dataloader: DataLoader
            desc: Description for progress bar
            
        Returns:
            all_scores: List of anomaly scores
            all_labels: List of ground truth labels
            all_patch_scores: List of patch-wise scores
            all_paths: List of image paths
        """
        self.model.eval()
        
        all_scores = []
        all_labels = []
        all_patch_scores = []
        all_paths = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc):
                patches = batch['patches'].to(self.config.DEVICE)
                context_indices = batch['context_indices'].to(self.config.DEVICE)
                target_indices = batch['target_indices'].to(self.config.DEVICE)
                labels = batch['label'].numpy()
                paths = batch['path']
                
                # Forward pass
                predictions, targets, _ = self.model(
                    patches, context_indices, target_indices
                )
                
                # Compute anomaly scores
                scores = compute_anomaly_scores(predictions, targets)
                patch_scores = compute_patch_anomaly_scores(predictions, targets)
                
                all_scores.extend(scores.tolist())
                all_labels.extend(labels.tolist())
                all_patch_scores.extend(patch_scores.tolist())
                all_paths.extend(paths)
        
        return all_scores, all_labels, all_patch_scores, all_paths
    
    def fit_threshold(self, train_loader):
        """
        Fit adaptive threshold using training data (real images only)
        
        Args:
            train_loader: Training dataloader
        """
        print("\nFitting adaptive threshold on training data...")
        train_scores, _, _, _ = self.compute_scores(
            train_loader, 
            desc='Computing training scores'
        )
        
        self.threshold = self.threshold_computer.fit(train_scores)
        
        return self.threshold
    
    def predict(self, eval_loader):
        """
        Predict anomalies on evaluation data
        
        Args:
            eval_loader: Evaluation dataloader
            
        Returns:
            results: Dictionary containing predictions, scores, labels, etc.
        """
        if self.threshold is None:
            raise ValueError("Threshold not fitted. Call fit_threshold() first.")
        
        print("\nComputing anomaly scores on evaluation data...")
        eval_scores, eval_labels, eval_patch_scores, eval_paths = self.compute_scores(
            eval_loader,
            desc='Evaluating'
        )
        
        # Predict using threshold
        predictions = self.threshold_computer.predict(eval_scores)
        
        # Compute metrics
        metrics = compute_metrics(predictions, eval_labels)
        
        # Separate scores by class
        eval_scores = np.array(eval_scores)
        eval_labels = np.array(eval_labels)
        
        real_scores = eval_scores[eval_labels == 0]
        fake_scores = eval_scores[eval_labels == 1]
        
        results = {
            'predictions': predictions,
            'scores': eval_scores,
            'labels': eval_labels,
            'patch_scores': eval_patch_scores,
            'paths': eval_paths,
            'threshold': self.threshold,
            'metrics': metrics,
            'real_scores': real_scores,
            'fake_scores': fake_scores
        }
        
        return results
    
    def explain(self, image, patches, context_indices, target_indices):
        """
        Generate explanation for a single image
        
        Args:
            image: Input image tensor [C, H, W]
            patches: Image patches [N, C, H, W]
            context_indices: Context patch indices
            target_indices: Target patch indices
            
        Returns:
            anomaly_score: Overall anomaly score
            patch_scores: Per-patch anomaly scores
            most_anomalous_patch_idx: Index of most anomalous patch
        """
        self.model.eval()
        
        with torch.no_grad():
            # Add batch dimension
            patches = patches.unsqueeze(0).to(self.config.DEVICE)
            context_indices = context_indices.unsqueeze(0).to(self.config.DEVICE)
            target_indices = target_indices.unsqueeze(0).to(self.config.DEVICE)
            
            # Forward pass
            predictions, targets, _ = self.model(
                patches, context_indices, target_indices
            )
            
            # Compute scores
            anomaly_score = compute_anomaly_scores(predictions, targets)[0]
            patch_scores = compute_patch_anomaly_scores(predictions, targets)[0]
            
            # Find most anomalous patch
            most_anomalous_patch_idx = target_indices[0][np.argmax(patch_scores)].item()
        
        return anomaly_score, patch_scores, most_anomalous_patch_idx


def evaluate(config, checkpoint_path, visualize_samples=True):
    """
    Main evaluation function
    
    Args:
        config: Configuration object
        checkpoint_path: Path to trained model checkpoint
        visualize_samples: Whether to visualize sample predictions
    """
    # Create directories
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    # Initialize logger
    logger = Logger(os.path.join(config.LOG_DIR, 'evaluation.log'))
    
    logger.log("=" * 80)
    logger.log("CNN-JEPA Evaluation for Deepfake Detection")
    logger.log("=" * 80)
    
    # Create dataloaders
    logger.log("\nCreating dataloaders...")
    train_loader, eval_loader, patch_sampler = create_dataloaders(config)
    logger.log(f"Training samples: {len(train_loader.dataset)}")
    logger.log(f"Evaluation samples: {len(eval_loader.dataset)}")
    
    # Load model
    logger.log(f"\nLoading model from: {checkpoint_path}")
    model = CNNJEPA(config).to(config.DEVICE)
    
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.log("Model loaded successfully")
    
    # Create anomaly detector
    detector = AnomalyDetector(model, config)
    
    # Fit threshold on training data
    threshold = detector.fit_threshold(train_loader)
    
    # Predict on evaluation data
    results = detector.predict(eval_loader)
    
    # Print metrics
    logger.log("\n" + "=" * 80)
    logger.log("EVALUATION RESULTS")
    logger.log("=" * 80)
    logger.log(f"Threshold: {results['threshold']:.6f}")
    logger.log(f"\nMetrics:")
    logger.log(f"  Accuracy:  {results['metrics']['accuracy']:.4f}")
    logger.log(f"  Precision: {results['metrics']['precision']:.4f}")
    logger.log(f"  Recall:    {results['metrics']['recall']:.4f}")
    logger.log(f"  F1 Score:  {results['metrics']['f1']:.4f}")
    
    # Compute additional statistics
    logger.log(f"\nScore Statistics:")
    logger.log(f"  Real images:")
    logger.log(f"    Mean: {results['real_scores'].mean():.6f}")
    logger.log(f"    Std:  {results['real_scores'].std():.6f}")
    logger.log(f"    Min:  {results['real_scores'].min():.6f}")
    logger.log(f"    Max:  {results['real_scores'].max():.6f}")
    logger.log(f"  Fake images:")
    logger.log(f"    Mean: {results['fake_scores'].mean():.6f}")
    logger.log(f"    Std:  {results['fake_scores'].std():.6f}")
    logger.log(f"    Min:  {results['fake_scores'].min():.6f}")
    logger.log(f"    Max:  {results['fake_scores'].max():.6f}")
    
    # Save results
    results_path = os.path.join(config.RESULTS_DIR, 'evaluation_results.json')
    save_results(results, results_path)
    
    # Plot ROC curve
    logger.log("\nGenerating ROC curve...")
    roc_path = os.path.join(config.RESULTS_DIR, 'roc_curve.png')
    auc = plot_roc_curve(results['labels'], results['scores'], save_path=roc_path)
    logger.log(f"ROC AUC: {auc:.4f}")
    logger.log(f"ROC curve saved: {roc_path}")
    
    # Plot score distribution
    logger.log("\nGenerating score distribution plot...")
    dist_path = os.path.join(config.RESULTS_DIR, 'score_distribution.png')
    plot_score_distribution(
        results['real_scores'],
        results['fake_scores'],
        results['threshold'],
        save_path=dist_path
    )
    logger.log(f"Score distribution saved: {dist_path}")
    
    # Visualize sample predictions with explainability
    if visualize_samples and config.SAVE_ANOMALY_MAPS:
        logger.log("\nGenerating anomaly maps for sample images...")
        
        vis_dir = os.path.join(config.RESULTS_DIR, 'anomaly_maps')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Get some samples (mix of correct and incorrect predictions)
        all_indices = np.arange(len(results['predictions']))
        
        # Correct predictions
        correct_mask = results['predictions'] == results['labels']
        correct_indices = all_indices[correct_mask]
        
        # Incorrect predictions
        incorrect_mask = ~correct_mask
        incorrect_indices = all_indices[incorrect_mask]
        
        # Sample from both
        num_correct = min(config.VIS_NUM_SAMPLES // 2, len(correct_indices))
        num_incorrect = min(config.VIS_NUM_SAMPLES // 2, len(incorrect_indices))
        
        sample_indices = []
        if num_correct > 0:
            sample_indices.extend(np.random.choice(correct_indices, num_correct, replace=False))
        if num_incorrect > 0:
            sample_indices.extend(np.random.choice(incorrect_indices, num_incorrect, replace=False))
        
        for idx in sample_indices[:config.VIS_NUM_SAMPLES]:
            # Get sample data
            sample = eval_loader.dataset[idx]
            image = sample['image']
            patches = sample['patches']
            target_indices_sample = sample['target_indices']
            positions = sample['positions']
            label = sample['label']
            path = sample['path']
            
            # Get prediction
            pred = results['predictions'][idx]
            score = results['scores'][idx]
            patch_scores = results['patch_scores'][idx]
            
            # Create visualization
            save_name = f"sample_{idx}_true_{label}_pred_{pred}_score_{score:.4f}.png"
            save_path = os.path.join(vis_dir, save_name)
            
            visualize_anomaly_map(
                image,
                patch_scores,
                [positions[i] for i in target_indices_sample.numpy()],
                save_path=save_path
            )
        
        logger.log(f"Anomaly maps saved to: {vis_dir}")
    
    logger.log("\n" + "=" * 80)
    logger.log("Evaluation completed!")
    logger.log("=" * 80)
    
    return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Evaluate CNN-JEPA for Deepfake Detection')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--no_vis', action='store_true',
                       help='Disable visualization of sample predictions')
    parser.add_argument('--num_vis', type=int, default=None,
                       help='Number of samples to visualize (overrides config)')
    
    args = parser.parse_args()
    
    # Override config if specified
    if args.num_vis:
        Config.VIS_NUM_SAMPLES = args.num_vis
    
    # Evaluate
    evaluate(Config, args.checkpoint, visualize_samples=not args.no_vis)


if __name__ == "__main__":
    main()
