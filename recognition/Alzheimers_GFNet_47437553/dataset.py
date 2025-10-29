"""
ADNI Dataset Loader for Alzheimer's Disease Classification

This module provides data loading utilities for the ADNI (Alzheimer's Disease Neuroimaging Initiative)
dataset, including train/validation split, data augmentation, and balanced sampling.

Classes:
- ADNIDataset: PyTorch Dataset for loading brain scan images
- Functions for building data pipelines with preprocessing and augmentation

Usage:
    from dataset import build_data_pipeline
    data_loaders = build_data_pipeline(batch_size=32, val_fraction=0.2)
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
import random
from typing import Dict, Tuple, Optional, List
from sklearn.model_selection import train_test_split


class ADNIDataset(Dataset):
    """PyTorch Dataset for ADNI brain scan images (JPEG format).
    Supports train/test modes with configurable augmentation and transforms."""
    
    def __init__(self, data_root='/home/groups/comp3710/ADNI/AD_NC', 
                 mode='train', transform=None, augment=True):
        """Initialize ADNI dataset with specified mode and transformations.
        Args: data_root: Path to dataset root, mode: 'train' or 'test', 
        transform: Custom transforms, augment: Enable data augmentation"""
        self.data_root = Path(data_root)
        self.mode = mode
        self.augment = augment
        
        self.samples = []
        self.labels = []
        
        # Class mapping: 0=Normal Cognitive, 1=Alzheimer's Disease
        class_dirs = {
            'NC': 0,
            'AD': 1
        }
        
        mode_path = self.data_root / mode
        
        if not mode_path.exists():
            raise FileNotFoundError(f"Mode path does not exist: {mode_path}")
        
        print(f"Loading data from: {mode_path}")
        
        # Scan directories for images
        for class_name, label in class_dirs.items():
            class_path = mode_path / class_name
            if class_path.exists():
                print(f"  Searching in {class_path}")
                
                jpg_files = list(class_path.glob('*.jpg')) + list(class_path.glob('*.jpeg'))
                print(f"  Found {len(jpg_files)} images for class {class_name}")
                
                for img_path in jpg_files:
                    self.samples.append(str(img_path))
                    self.labels.append(label)
            else:
                print(f"  Warning: Class path does not exist: {class_path}")
        
        # Convert to numpy arrays with explicit dtypes for consistency
        self.samples = np.array(self.samples, dtype=str)
        self.labels = np.array(self.labels, dtype=np.int64)
        
        if len(self.samples) == 0:
            raise ValueError(f"No samples found in {mode_path}. Check directory structure.")
        
        print(f"Found {len(self.samples)} total samples")
        print(f"Class distribution: CN={np.sum(self.labels==0)}, AD={np.sum(self.labels==1)}")
        
        # Configure transforms based on mode and augmentation settings
        if transform is None:
            if mode == 'train' and augment:
                # Training augmentation: rotation, flips, color jitter
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomRotation(15),
                    transforms.RandomAffine(
                        degrees=0,
                        translate=(0.1, 0.1),
                        scale=(0.9, 1.1)
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                # Validation/test: only resize and normalize
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform
    
    def __len__(self):
        """Return total number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Load and transform a single sample.
        Returns: tuple of (image_tensor, label_tensor)"""
        image_path = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return image, label


def build_data_pipeline(batch_size=16, num_workers=4, 
                       val_fraction=0.2, augment=True, 
                       data_root='/home/groups/comp3710/ADNI/AD_NC') -> Dict[str, DataLoader]:
    """Build complete data pipeline with stratified train/val split and balanced sampling.
    Returns dictionary containing 'train' and 'val' DataLoaders with appropriate preprocessing."""
    
    # Load full training dataset
    full_dataset = ADNIDataset(
        data_root=data_root,
        mode='train',
        augment=False
    )
    
    # Stratified split to maintain class distribution
    indices = np.arange(len(full_dataset))
    labels = full_dataset.labels
    
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_fraction,
        stratify=labels,
        random_state=42
    )
    
    print(f"\nDataset split:")
    print(f"  Training samples: {len(train_idx)}")
    print(f"  Validation samples: {len(val_idx)}")
    
    # Training transforms with comprehensive augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomApply([transforms.ColorJitter(0.15, 0.15, 0.1, 0.03)], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3))
    ])
    
    # Validation transforms without augmentation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset subsets with appropriate transforms
    train_dataset = ADNIDataset(
        data_root=data_root,
        mode='train',
        transform=train_transform,
        augment=False
    )
    train_dataset = torch.utils.data.Subset(train_dataset, train_idx)
    
    val_dataset = ADNIDataset(
        data_root=data_root,
        mode='train',
        transform=val_transform,
        augment=False
    )
    val_dataset = torch.utils.data.Subset(val_dataset, val_idx)
    
    # Calculate inverse frequency weights for balanced sampling
    train_labels = labels[train_idx]
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_labels]
    
    print(f"\nClass weights: CN={class_weights[0]:.3f}, AD={class_weights[1]:.3f}")
    
    # Weighted sampler ensures balanced batches during training
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_idx),
        replacement=True
    )
    
    # Create DataLoaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader
    }


def get_test_loader(batch_size=16, num_workers=4,
                   data_root='/home/groups/comp3710/ADNI/AD_NC') -> DataLoader:
    """Create DataLoader for test set without augmentation.
    Returns: DataLoader for test dataset with standard preprocessing only."""
    test_dataset = ADNIDataset(
        data_root=data_root,
        mode='test',
        augment=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return test_loader


if __name__ == '__main__':
    """Test script to verify data pipeline functionality and data loading."""
    print("Testing ADNI data pipeline...")
    print("="*70)
    
    try:
        print("\n1. Testing train dataset loading...")
        data_loaders = build_data_pipeline(batch_size=4, num_workers=0)
        train_loader = data_loaders['train']
        val_loader = data_loaders['val']
        
        print(f"\n✓ Train batches: {len(train_loader)}")
        print(f"✓ Val batches: {len(val_loader)}")
        
        # Verify batch loading and data format
        print("\n2. Testing batch loading...")
        for images, labels in train_loader:
            print(f"✓ Batch shape: {images.shape}")
            print(f"✓ Labels: {labels}")
            print(f"✓ Label dtype: {labels.dtype}")
            print(f"✓ Image stats - Min: {images.min():.3f}, Max: {images.max():.3f}, "
                  f"Mean: {images.mean():.3f}, Std: {images.std():.3f}")
            break
        
        # Test test set loading
        print("\n3. Testing test dataset loading...")
        test_loader = get_test_loader(batch_size=4, num_workers=0)
        print(f"✓ Test batches: {len(test_loader)}")
        
        for images, labels in test_loader:
            print(f"✓ Test batch shape: {images.shape}")
            print(f"✓ Test labels: {labels}")
            break
            
        print("\n" + "="*70)
        print("✓ Data pipeline test successful!")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ Error in data pipeline: {e}")
        import traceback
        traceback.print_exc()