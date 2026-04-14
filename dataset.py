"""
Dataset loader for CNN-JEPA Deepfake Detection
Handles image loading, preprocessing, and patch splitting
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import random
from config import Config


class PatchSampler:
    """
    Handles patch extraction and sampling for region-aware JEPA
    Splits images into grid patches and samples context/target subsets
    """
    
    def __init__(self, image_size, patch_grid_size, context_ratio):
        """
        Args:
            image_size: Size of input image (assumes square)
            patch_grid_size: Grid size (e.g., 4 for 4x4 grid)
            context_ratio: Ratio of patches to use as context
        """
        self.image_size = image_size
        self.patch_grid_size = patch_grid_size
        self.num_patches = patch_grid_size ** 2
        self.patch_size = image_size // patch_grid_size
        self.context_ratio = context_ratio
        
        # Calculate number of context and target patches
        self.num_context = int(self.num_patches * context_ratio)
        self.num_target = self.num_patches - self.num_context
        
    def extract_patches(self, image):
        """
        Extract patches from image in grid format
        
        Args:
            image: Tensor of shape [C, H, W]
            
        Returns:
            patches: Tensor of shape [num_patches, C, patch_size, patch_size]
            positions: List of (row, col) positions for each patch
        """
        C, H, W = image.shape
        assert H == W == self.image_size, f"Image size mismatch: {H}x{W} vs {self.image_size}"
        
        patches = []
        positions = []
        
        for i in range(self.patch_grid_size):
            for j in range(self.patch_grid_size):
                # Extract patch
                row_start = i * self.patch_size
                col_start = j * self.patch_size
                patch = image[:, 
                            row_start:row_start + self.patch_size,
                            col_start:col_start + self.patch_size]
                
                patches.append(patch)
                positions.append((i, j))
        
        patches = torch.stack(patches, dim=0)  # [num_patches, C, patch_size, patch_size]
        return patches, positions
    
    def sample_context_target(self, num_patches):
        """
        Randomly sample which patches are context vs target
        
        Args:
            num_patches: Total number of patches
            
        Returns:
            context_indices: List of indices for context patches
            target_indices: List of indices for target patches
        """
        all_indices = list(range(num_patches))
        random.shuffle(all_indices)
        
        context_indices = all_indices[:self.num_context]
        target_indices = all_indices[self.num_context:]
        
        return context_indices, target_indices
    
    def create_mask(self, context_indices, target_indices):
        """
        Create binary masks for context and target patches
        
        Args:
            context_indices: Indices of context patches
            target_indices: Indices of target patches
            
        Returns:
            context_mask: Binary mask [num_patches]
            target_mask: Binary mask [num_patches]
        """
        context_mask = torch.zeros(self.num_patches)
        target_mask = torch.zeros(self.num_patches)
        
        context_mask[context_indices] = 1.0
        target_mask[target_indices] = 1.0
        
        return context_mask, target_mask


class DeepfakeDataset(Dataset):
    """
    Dataset for loading real/fake images for JEPA training and evaluation
    During training (self-supervised): only uses REAL images
    During evaluation: uses both REAL and FAKE images
    """
    
    def __init__(self, root_dir, mode='train', transform=None):
        """
        Args:
            root_dir: Root directory containing 'real' and 'fake' subdirectories
            mode: 'train' (only real), 'eval' (real + fake)
            transform: Torchvision transforms to apply
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        
        # Load image paths
        self.image_paths = []
        self.labels = []  # 0 = real, 1 = fake
        
        # Load real images
        real_dir = os.path.join(root_dir, 'real')
        if os.path.exists(real_dir):
            real_images = [f for f in os.listdir(real_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            for img in real_images:
                self.image_paths.append(os.path.join(real_dir, img))
                self.labels.append(0)  # Real
        
        # Load fake images (only in eval mode)
        if mode == 'eval':
            fake_dir = os.path.join(root_dir, 'fake')
            if os.path.exists(fake_dir):
                fake_images = [f for f in os.listdir(fake_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                for img in fake_images:
                    self.image_paths.append(os.path.join(fake_dir, img))
                    self.labels.append(1)  # Fake
        
        print(f"Loaded {len(self.image_paths)} images in {mode} mode")
        print(f"  - Real: {self.labels.count(0)}")
        print(f"  - Fake: {self.labels.count(1)}")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Load and return image with label
        
        Returns:
            image: Transformed image tensor
            label: 0 (real) or 1 (fake)
            path: Original image path
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_path


class JEPADataset(Dataset):
    """
    Wrapper dataset for JEPA training with patch sampling
    Returns image, patches, context/target indices
    """
    
    def __init__(self, base_dataset, patch_sampler):
        """
        Args:
            base_dataset: Base DeepfakeDataset
            patch_sampler: PatchSampler instance
        """
        self.base_dataset = base_dataset
        self.patch_sampler = patch_sampler
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        """
        Returns image with extracted patches and sampling indices
        """
        image, label, path = self.base_dataset[idx]
        
        # Extract patches
        patches, positions = self.patch_sampler.extract_patches(image)
        
        # Sample context and target patches
        context_indices, target_indices = self.patch_sampler.sample_context_target(
            len(patches)
        )
        
        # Create masks
        context_mask, target_mask = self.patch_sampler.create_mask(
            context_indices, target_indices
        )
        
        return {
            'image': image,
            'patches': patches,
            'context_indices': torch.tensor(context_indices, dtype=torch.long),
            'target_indices': torch.tensor(target_indices, dtype=torch.long),
            'context_mask': context_mask,
            'target_mask': target_mask,
            'positions': positions,
            'label': label,
            'path': path
        }


def get_transforms(mode='train'):
    """
    Get image transformations for training/evaluation
    
    Args:
        mode: 'train' or 'eval'
        
    Returns:
        transforms: Torchvision transforms
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


def create_dataloaders(config):
    """
    Create dataloaders for training and evaluation
    
    Args:
        config: Configuration object
        
    Returns:
        train_loader: DataLoader for training (only real images)
        eval_loader: DataLoader for evaluation (real + fake images)
    """
    # Create patch sampler
    patch_sampler = PatchSampler(
        image_size=config.IMAGE_SIZE,
        patch_grid_size=config.PATCH_GRID_SIZE,
        context_ratio=config.CONTEXT_RATIO
    )
    
    # Create base datasets
    train_dataset = DeepfakeDataset(
        root_dir=config.DATASET_ROOT,
        mode='train',  # Only real images
        transform=get_transforms('train')
    )
    
    eval_dataset = DeepfakeDataset(
        root_dir=config.DATASET_ROOT,
        mode='eval',  # Real + fake images
        transform=get_transforms('eval')
    )
    
    # Wrap in JEPA datasets
    train_jepa_dataset = JEPADataset(train_dataset, patch_sampler)
    eval_jepa_dataset = JEPADataset(eval_dataset, patch_sampler)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_jepa_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True if config.DEVICE.type == 'cuda' else False
    )
    
    eval_loader = DataLoader(
        eval_jepa_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True if config.DEVICE.type == 'cuda' else False
    )
    
    return train_loader, eval_loader, patch_sampler


if __name__ == "__main__":
    # Test dataset loading
    print("Testing dataset loading...")
    
    # Create dummy dataset structure
    os.makedirs("dataset/real", exist_ok=True)
    os.makedirs("dataset/fake", exist_ok=True)
    
    train_loader, eval_loader, patch_sampler = create_dataloaders(Config)
    
    print(f"\nTrain loader: {len(train_loader)} batches")
    print(f"Eval loader: {len(eval_loader)} batches")
    
    # Test one batch
    if len(train_loader) > 0:
        batch = next(iter(train_loader))
        print(f"\nBatch keys: {batch.keys()}")
        print(f"Image shape: {batch['image'].shape}")
        print(f"Patches shape: {batch['patches'].shape}")
        print(f"Context indices: {batch['context_indices'].shape}")
        print(f"Target indices: {batch['target_indices'].shape}")
