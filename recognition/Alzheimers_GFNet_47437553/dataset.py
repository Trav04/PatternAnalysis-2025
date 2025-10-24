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

