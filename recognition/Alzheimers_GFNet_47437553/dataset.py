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
    """
    ADNI Dataset for JPEG brain scan images
    """
    
    def __init__(self, data_root='/home/groups/comp3710/ADNI/AD_NC', 
                 mode='train', transform=None, augment=True):
        """
        Args:
            data_root: Path to ADNI dataset (contains train/ and test/ subdirs)
            mode: 'train' or 'test'
            transform: torchvision transforms
            augment: Apply data augmentation
        """
        self.data_root = Path(data_root)
        self.mode = mode
        self.augment = augment
        
        # Collect all image paths and labels
        self.samples = []
        self.labels = []
        
        # Class directories
        class_dirs = {
            'NC': 0,  # Cognitively Normal
            'AD': 1   # Alzheimer's Disease
        }
        
        # Look in train/ or test/ subdirectory based on mode
        mode_path = self.data_root / mode
        
        if not mode_path.exists():
            raise FileNotFoundError(f"Mode path does not exist: {mode_path}")
        
        print(f"Loading data from: {mode_path}")
        
        for class_name, label in class_dirs.items():
            class_path = mode_path / class_name
            if class_path.exists():
                print(f"  Searching in {class_path}")
                
                # Look for JPEG files
                jpg_files = list(class_path.glob('*.jpg')) + list(class_path.glob('*.jpeg'))
                print(f"  Found {len(jpg_files)} images for class {class_name}")
                
                for img_path in jpg_files:
                    self.samples.append(str(img_path))
                    self.labels.append(label)
            else:
                print(f"  Warning: Class path does not exist: {class_path}")
        
        # Convert to arrays with explicit dtype
        self.samples = np.array(self.samples, dtype=str)
        self.labels = np.array(self.labels, dtype=np.int64)
        
        if len(self.samples) == 0:
            raise ValueError(f"No samples found in {mode_path}. Check directory structure.")
        
        print(f"Found {len(self.samples)} total samples")
        print(f"Class distribution: CN={np.sum(self.labels==0)}, AD={np.sum(self.labels==1)}")
        
        # Setup transforms
        if transform is None:
            if mode == 'train' and augment:
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
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label as long tensor
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return image, label


def build_data_pipeline(batch_size=16, num_workers=4, 
                       val_fraction=0.2, augment=True, 
                       data_root='/home/groups/comp3710/ADNI/AD_NC') -> Dict[str, DataLoader]:
    """
    Build complete data pipeline with train/val split
    
    Returns:
        Dictionary with 'train' and 'val' DataLoaders
    """
    
    # Load full training dataset
    full_dataset = ADNIDataset(
        data_root=data_root,
        mode='train',
        augment=False  # We'll handle augmentation separately
    )
    
    # Split into train and validation
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
    
    # Create train transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(0.9,1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomApply([transforms.ColorJitter(0.15,0.15,0.1,0.03)], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # or your computed mean
                            std=[0.229, 0.224, 0.225]),  # or your computed std
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3))
    ])
    
    # Create validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create train dataset with augmentation
    train_dataset = ADNIDataset(
        data_root=data_root,
        mode='train',
        transform=train_transform,
        augment=False  # Augmentation is in transform
    )
    train_dataset = torch.utils.data.Subset(train_dataset, train_idx)
    
    # Create validation dataset without augmentation
    val_dataset = ADNIDataset(
        data_root=data_root,
        mode='train',
        transform=val_transform,
        augment=False
    )
    val_dataset = torch.utils.data.Subset(val_dataset, val_idx)
    
    # Calculate sample weights for balanced training
    train_labels = labels[train_idx]
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_labels]
    
    print(f"\nClass weights: CN={class_weights[0]:.3f}, AD={class_weights[1]:.3f}")
    
    # Create weighted sampler for training
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_idx),
        replacement=True
    )
    
    # Create data loaders
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
    """
    Get test data loader
    """
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
    # Test the data pipeline
    print("Testing ADNI data pipeline...")
    print("="*70)
    
    try:
        # Test with train mode
        print("\n1. Testing train dataset loading...")
        data_loaders = build_data_pipeline(batch_size=4, num_workers=0)
        train_loader = data_loaders['train']
        val_loader = data_loaders['val']
        
        print(f"\n✓ Train batches: {len(train_loader)}")
        print(f"✓ Val batches: {len(val_loader)}")
        
        # Test loading a batch
        print("\n2. Testing batch loading...")
        for images, labels in train_loader:
            print(f"✓ Batch shape: {images.shape}")
            print(f"✓ Labels: {labels}")
            print(f"✓ Label dtype: {labels.dtype}")
            print(f"✓ Image stats - Min: {images.min():.3f}, Max: {images.max():.3f}, "
                  f"Mean: {images.mean():.3f}, Std: {images.std():.3f}")
            break
        
        # Test test loader
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