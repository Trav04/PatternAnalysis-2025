import torch
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms
from typing import Tuple, Dict, List
import numpy as np

class ADNIBrainDataset:
  """
  Using preprocessed images provided by the ADNI brain dataset.
  """
    def __init__(self):
        self.base_directory = Path('/home/groups/comp3710/ADNI/AD_NC')
        self.image_dimensions = 224
        self.mean_normalization = 0.0062
        self.std_normalization = 0.0083
        self.rotation_range = 10
        self.brightness_min = 0.8
        self.brightness_max = 1.2
        
    def build_train_transforms(self):
      """
      Transforms the data for the training set 
      """
        return transforms.Compose([
            transforms.RandomRotation(degrees=self.rotation_range),
            transforms.RandomResizedCrop(size=self.image_dimensions),
            transforms.ColorJitter(brightness=(self.brightness_min, self.brightness_max)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[self.mean_normalization], std=[self.std_normalization])
        ])
    
    def build_eval_transforms(self):
      """
      Transforms the data for evaluation test set
      """
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.image_dimensions),
            transforms.ToTensor(),
            transforms.Normalize(mean=[self.mean_normalization], std=[self.std_normalization])
        ])

class BrainScanLoader(Dataset):
    """
    Custom PyTorch Dataset class for loading brain scan images from the ADNI dataset.
    Handles both Alzheimer's Disease (AD) and Normal Control (NC) classes.
    """
    
    CLASS_MAPPING = {'AD': 1, 'NC': 0}
    
    def __init__(self, config: BrainImageConfig, mode: str = 'train', 
                 transform_pipeline=None):
        """
        Initializes the dataset by scanning the directory structure and collecting
        all image paths with their corresponding labels.
        """
        self.config = config
        self.mode = mode
        self.transform_pipeline = transform_pipeline
        self.samples = self._scan_directory()
        
    def _scan_directory(self) -> List[Tuple[Path, int]]:
        """
        Scans the dataset directory and collects all image file paths along with
        their class labels based on the folder structure.
        """
        subset_path = self.config.base_directory / self.mode
        collected_samples = []
        
        for class_name, label_value in self.CLASS_MAPPING.items():
            class_folder = subset_path / class_name
            if class_folder.exists():
                image_files = sorted(class_folder.glob('*'))
                for img_file in image_files:
                    collected_samples.append((img_file, label_value))
        
        return collected_samples
    
    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Loads and returns a single brain scan image and its label at the given index.
        Applies transformations if specified.
        """
        img_path, target = self.samples[idx]
        brain_image = Image.open(img_path).convert('L')
        
        if self.transform_pipeline:
            brain_image = self.transform_pipeline(brain_image)
            
        return brain_image, target