"""
Prediction Script for GFNet-based Alzheimer's Disease Classification

This script demonstrates how to use the trained enhanced GFNet model to classify
brain MRI images as Normal Cognitive (NC) or Alzheimer's Disease (AD).

Features:
- Loads the trained model from checkpoint
- Performs inference on test samples with Test-Time Augmentation (TTA)
- Visualizes predictions with confidence scores
- Prints comprehensive performance metrics
- Generates publication-quality prediction visualization

Usage:
    python predict.py

Requirements:
    - PyTorch with CUDA support (recommended)
    - modules.py
    - dataset.py
    - matplotlib, numpy, scikit-learn, seaborn
    - enhanced_gfnet_best_model.pth (trained model checkpoint)
"""

import torch
import torch.nn.functional as F
from torch import amp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import os

# Import model and dataset modules
from modules import create_enhanced_gfnet_base
from dataset import build_data_pipeline


class ADNIPredictor:
    """Predictor class for Alzheimer's Disease classification using GFNet"""
    
    def __init__(self, model_path='enhanced_gfnet_best_model.pth', device=None):
        """
        Initialize the predictor with trained model
        
        Args:
            model_path (str): Path to the trained model checkpoint
            device (torch.device): Device to run inference on (cuda/cpu)
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = ['Normal Cognitive (NC)', 'Alzheimer\'s Disease (AD)']
        
        # Initialize model architecture
        print("Initializing GFNet model architecture...")
        self.model = create_enhanced_gfnet_base(num_classes=2, img_size=224, in_chans=3)
        
        # Load trained weights
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
        
        print(f"Loading trained model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    def predict_with_tta(self, images, tta_transforms=None):
        """
        Perform prediction with Test-Time Augmentation
        
        Args:
            images (torch.Tensor): Input images (batch_size, 3, 224, 224)
            tta_transforms (list): List of augmentation functions
            
        Returns:
            tuple: (probabilities, predictions)
        """
        if tta_transforms is None:
            # Default TTA: original + horizontal flip
            tta_transforms = [
                lambda x: x,  # Original
                lambda x: torch.flip(x, dims=[3])  # Horizontal flip
            ]
        
        images = images.to(self.device)
        batch_logits = []
        
        with torch.no_grad():
            for transform in tta_transforms:
                aug_images = transform(images)
                with amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    outputs = self.model(aug_images)  # Get logits
                    batch_logits.append(outputs)
        
        # Average logits across augmentations (proper TTA ensemble)
        avg_logits = torch.stack(batch_logits).mean(dim=0)
        
        # Convert to probabilities
        probabilities = F.softmax(avg_logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        
        return probabilities, predictions
    
    def evaluate_test_set(self, test_loader):
        """
        Evaluate model performance on entire test set
        
        Args:
            test_loader (DataLoader): Test data loader
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        print("\n" + "="*70)
        print("Evaluating model on test set...")
        print("="*70)
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                # Predict with TTA
                probs, preds = self.predict_with_tta(images)
                
                all_preds.append(preds.cpu())
                all_labels.append(labels)
                all_probs.append(probs.cpu())
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Processed {batch_idx + 1}/{len(test_loader)} batches")
        
        # Concatenate results
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        all_probs = torch.cat(all_probs, dim=0).numpy()
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Classification report
        report = classification_report(
            all_labels, 
            all_preds, 
            target_names=self.class_names,
            digits=4
        )
        
        print("\n" + "="*70)
        print("TEST SET PERFORMANCE")
        print("="*70)
        print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("\nClassification Report:")
        print(report)
        print("\nConfusion Matrix:")
        print(f"{'':20s} {'Predicted NC':15s} {'Predicted AD':15s}")
        print(f"{'Actual NC':20s} {cm[0,0]:15d} {cm[0,1]:15d}")
        print(f"{'Actual AD':20s} {cm[1,0]:15d} {cm[1,1]:15d}")
        print("="*70)
        
        return {
            'accuracy': accuracy,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs,
            'confusion_matrix': cm
        }
    
    def visualize_predictions(self, images, labels, predictions, probabilities, 
                             num_samples=9, save_path='prediction_results.png'):
        """
        Create a visualization grid showing model predictions
        
        Args:
            images (torch.Tensor): Input images
            labels (torch.Tensor): True labels
            predictions (torch.Tensor): Predicted labels
            probabilities (torch.Tensor): Prediction probabilities
            num_samples (int): Number of samples to display (must be perfect square)
            save_path (str): Path to save the visualization
        """
        # Ensure we have enough samples
        num_samples = min(num_samples, len(images))
        grid_size = int(np.sqrt(num_samples))
        num_samples = grid_size * grid_size  # Ensure perfect square
        
        # Convert to numpy for visualization
        images_np = images[:num_samples].cpu().numpy()
        labels_np = labels[:num_samples].cpu().numpy()
        preds_np = predictions[:num_samples].cpu().numpy()
        probs_np = probabilities[:num_samples].cpu().numpy()
        
        # Create figure
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        fig.suptitle('GFNet Alzheimer\'s Disease Classification Results', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        for idx in range(num_samples):
            row = idx // grid_size
            col = idx % grid_size
            ax = axes[row, col] if grid_size > 1 else axes
            
            # Get image (denormalize for visualization)
            img = images_np[idx].transpose(1, 2, 0)  # CHW to HWC
            # Denormalize (assuming ImageNet normalization)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            
            # Display image
            ax.imshow(img)
            
            # Get prediction details
            true_label = int(labels_np[idx])
            pred_label = int(preds_np[idx])
            confidence = probs_np[idx, pred_label] * 100
            
            # Determine if prediction is correct
            is_correct = (true_label == pred_label)
            color = 'green' if is_correct else 'red'
            
            # Create title
            true_class = 'NC' if true_label == 0 else 'AD'
            pred_class = 'NC' if pred_label == 0 else 'AD'
            
            title = f'Pred: {pred_class}, True: {true_class}\nConfidence: {confidence:.1f}%'
            ax.set_title(title, fontsize=10, color=color, fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPrediction visualization saved to: {save_path}")
        plt.show()
    
    def plot_confusion_matrix(self, cm, save_path='confusion_matrix.png'):
        """
        Plot confusion matrix as a heatmap
        
        Args:
            cm (np.ndarray): Confusion matrix
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['NC', 'AD'],
                   yticklabels=['NC', 'AD'],
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - GFNet AD Classification', 
                 fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
        plt.show()


def main():
    """Main function to run predictions and visualizations"""
    
    print("="*70)
    print("GFNet Alzheimer's Disease Classification - Prediction Demo")
    print("="*70)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Build data pipeline
    print("\nLoading test dataset...")
    data_loaders = build_data_pipeline(batch_size=32, val_fraction=0.2, augment=False)
    
    # Use validation set as test set for demonstration
    test_loader = data_loaders['val']
    print(f"Test dataset size: {len(test_loader.dataset)} samples")
    
    # Initialize predictor
    predictor = ADNIPredictor(model_path='enhanced_gfnet_best_model.pth')
    
    # Evaluate on entire test set
    results = predictor.evaluate_test_set(test_loader)
    
    # Get a batch of samples for visualization
    print("\nGenerating prediction visualizations...")
    images, labels = next(iter(test_loader))
    
    # Make predictions on this batch
    probs, preds = predictor.predict_with_tta(images)
    
    # Visualize predictions (9 samples in 3x3 grid)
    predictor.visualize_predictions(
        images=images,
        labels=labels,
        predictions=preds,
        probabilities=probs,
        num_samples=9,
        save_path='prediction_results.png'
    )
    
    # Plot confusion matrix
    predictor.plot_confusion_matrix(
        cm=results['confusion_matrix'],
        save_path='confusion_matrix.png'
    )
    
    print("\n" + "="*70)
    print("Prediction demo completed successfully!")
    print("="*70)
    print("\nGenerated files:")
    print("  - prediction_results.png: Visual predictions on sample images")
    print("  - confusion_matrix.png: Confusion matrix heatmap")
    print("\nModel Summary:")
    print(f"  - Test Accuracy: {results['accuracy']*100:.2f}%")
    print(f"  - Test Samples: {len(results['labels'])}")
    print(f"  - Architecture: Enhanced GFNet Base")
    print(f"  - Input Size: 224x224x3")
    print(f"  - Classes: {predictor.class_names}")
    print("="*70)


if __name__ == '__main__':
    main()