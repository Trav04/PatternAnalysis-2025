"""
Training Script for GFNet-based Alzheimer's Disease Classification

This script trains a Global Filter Network (GFNet) model to classify brain MRI scans
as either Normal Control (NC) or Alzheimer's Disease (AD). It includes:
- Training and validation loops
- Learning rate scheduling
- Metrics tracking and visualization
- Model checkpointing
- Progress monitoring

Usage:
    python train.py

Requirements:
    - PyTorch with CUDA support (recommended)
    - modules.py (GFNet model implementation)
    - dataset.py (ADNI dataset loader)
    - matplotlib, numpy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time
from datetime import timedelta

# Import custom modules
from modules import gfnet_base, gfnet_small
from dataset import build_data_pipeline


def train_one_epoch(epoch, model, train_loader, criterion, optimizer, scheduler, 
                    train_losses, train_accuracies, device):
    """
    Train the model for one epoch with detailed progress tracking.
    
    Args:
        epoch (int): Current epoch number (0-indexed)
        model (nn.Module): GFNet model to train
        train_loader (DataLoader): DataLoader for training data
        criterion (nn.Module): Loss function (CrossEntropyLoss)
        optimizer (Optimizer): Optimizer for parameter updates
        scheduler (Scheduler): Learning rate scheduler
        train_losses (list): List to accumulate training losses
        train_accuracies (list): List to accumulate training accuracies
        device (torch.device): Device for computation (cuda/cpu)
    
    Returns:
        tuple: (average_loss, accuracy_percentage)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    print(f"\n{'='*70}")
    print(f"Epoch {epoch + 1} - Training Phase")
    print(f"{'='*70}")
    
    batch_count = len(train_loader)
    epoch_start = time.time()
    
    # Iterate through training batches
    for batch_idx, (images, labels) in enumerate(train_loader):
        batch_start = time.time()
        
        # Move data to device
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update learning rate (per-batch for warm restarts)
        scheduler.step(epoch + batch_idx / batch_count)
        
        # Calculate statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        batch_time = time.time() - batch_start
        current_acc = 100.0 * correct / total
        current_lr = scheduler.get_last_lr()[0]
        
        # Print progress every 10 batches or at the end
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == batch_count:
            print(f"  Batch [{batch_idx + 1:3d}/{batch_count:3d}] | "
                  f"Loss: {loss.item():.4f} | "
                  f"Acc: {current_acc:.2f}% | "
                  f"LR: {current_lr:.6f} | "
                  f"Time: {batch_time:.2f}s")
    
    # Calculate epoch statistics
    epoch_loss = running_loss / batch_count
    epoch_acc = 100.0 * correct / total
    epoch_time = time.time() - epoch_start
    
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    
    print(f"\n{'─'*70}")
    print(f"Training Summary:")
    print(f"  Average Loss: {epoch_loss:.4f}")
    print(f"  Accuracy: {epoch_acc:.2f}% ({correct}/{total})")
    print(f"  Epoch Time: {timedelta(seconds=int(epoch_time))}")
    print(f"{'─'*70}")
    
    return epoch_loss, epoch_acc


def validate_one_epoch(epoch, model, val_loader, criterion, val_losses, 
                       val_accuracies, device):
    """
    Validate the model for one epoch with detailed progress tracking.
    
    Args:
        epoch (int): Current epoch number (0-indexed)
        model (nn.Module): GFNet model to validate
        val_loader (DataLoader): DataLoader for validation data
        criterion (nn.Module): Loss function (CrossEntropyLoss)
        val_losses (list): List to accumulate validation losses
        val_accuracies (list): List to accumulate validation accuracies
        device (torch.device): Device for computation (cuda/cpu)
    
    Returns:
        tuple: (average_loss, accuracy_percentage)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Track per-class predictions for detailed analysis
    class_correct = [0, 0]  # [NC, AD]
    class_total = [0, 0]
    
    print(f"\n{'='*70}")
    print(f"Epoch {epoch + 1} - Validation Phase")
    print(f"{'='*70}")
    
    batch_count = len(val_loader)
    val_start = time.time()
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Calculate statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy tracking
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
            
            # Print progress every 5 batches or at the end
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == batch_count:
                current_acc = 100.0 * correct / total
                print(f"  Batch [{batch_idx + 1:3d}/{batch_count:3d}] | "
                      f"Loss: {loss.item():.4f} | "
                      f"Acc: {current_acc:.2f}%")
    
    # Calculate epoch statistics
    val_loss = running_loss / batch_count
    val_acc = 100.0 * correct / total
    val_time = time.time() - val_start
    
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    
    # Calculate per-class accuracies
    nc_acc = 100.0 * class_correct[0] / class_total[0] if class_total[0] > 0 else 0
    ad_acc = 100.0 * class_correct[1] / class_total[1] if class_total[1] > 0 else 0
    
    print(f"\n{'─'*70}")
    print(f"Validation Summary:")
    print(f"  Average Loss: {val_loss:.4f}")
    print(f"  Overall Accuracy: {val_acc:.2f}% ({correct}/{total})")
    print(f"  NC (Normal) Accuracy: {nc_acc:.2f}% ({class_correct[0]}/{class_total[0]})")
    print(f"  AD (Alzheimer's) Accuracy: {ad_acc:.2f}% ({class_correct[1]}/{class_total[1]})")
    print(f"  Validation Time: {timedelta(seconds=int(val_time))}")
    print(f"{'─'*70}")
    
    return val_loss, val_acc

def plot_metrics(num_epochs, train_losses, val_losses, train_accuracies, 
                val_accuracies, save_path='Training_vs_validation.png'):
    """
    Plot and save training and validation metrics.
    
    Args:
        num_epochs (int): Number of epochs trained
        train_losses (list): Training losses per epoch
        val_losses (list): Validation losses per epoch
        train_accuracies (list): Training accuracies per epoch
        val_accuracies (list): Validation accuracies per epoch
        save_path (str): Path to save the plot image
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    epochs = range(1, num_epochs + 1)
    
    # Plot losses
    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=4)
    ax1.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=4)
    ax1.set_title('Loss vs Epochs', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracies
    ax2.plot(epochs, train_accuracies, 'b-o', label='Train Accuracy', linewidth=2, markersize=4)
    ax2.plot(epochs, val_accuracies, 'r-s', label='Validation Accuracy', linewidth=2, markersize=4)
    ax2.axhline(y=80, color='g', linestyle='--', label='Target (80%)', linewidth=2)
    ax2.set_title('Accuracy vs Epochs', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Training plots saved to: {save_path}")
    plt.close()


def save_checkpoint(model, optimizer, scheduler, epoch, train_losses, val_losses,
                   train_accuracies, val_accuracies, filename='checkpoint.pth'):
    """
    Save model checkpoint with training state.
    
    Args:
        model (nn.Module): Model to save
        optimizer (Optimizer): Optimizer state
        scheduler (Scheduler): Scheduler state
        epoch (int): Current epoch number
        train_losses (list): Training loss history
        val_losses (list): Validation loss history
        train_accuracies (list): Training accuracy history
        val_accuracies (list): Validation accuracy history
        filename (str): Checkpoint filename
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
    }
    torch.save(checkpoint, filename)
    print(f"✓ Checkpoint saved to: {filename}")


def main():
    """
    Main training loop for GFNet Alzheimer's disease classification.
    """
    print("\n" + "="*70)
    print("GFNet Training for Alzheimer's Disease Classification")
    print("="*70 + "\n")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print()
    
    # Hyperparameters
    batch_size = 8
    base_lr = 0.0001
    num_epochs = 120
    weight_decay = 1e-4
    
    print("Hyperparameters:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {base_lr}")
    print(f"  Number of Epochs: {num_epochs}")
    print(f"  Weight Decay: {weight_decay}")
    print()
    
    # Load dataset
    print("Loading ADNI Dataset...")
    try:
        data_loaders = build_data_pipeline(batch_size=batch_size, mode='train', val_fraction=0.2)
        train_loader = data_loaders['train']
        val_loader = data_loaders['val']
        
        print(f"✓ Training samples: {len(train_loader.dataset)}")
        print(f"✓ Validation samples: {len(val_loader.dataset)}")
        print(f"✓ Training batches: {len(train_loader)}")
        print(f"✓ Validation batches: {len(val_loader)}")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return
    
    # Initialize model
    print("\nInitializing GFNet model...")
    try:
        # Using GFNet-Base architecture for balance of performance and efficiency
        model = gfnet_base(num_classes=2, img_size=224, in_chans=1)
        model = model.to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ Model: GFNet-Base")
        print(f"✓ Total Parameters: {total_params:,}")
        print(f"✓ Trainable Parameters: {trainable_params:,}")
    except Exception as e:
        print(f"✗ Error initializing model: {e}")
        return
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    
    # Learning rate scheduler with warm restarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,      # Initial restart interval
        T_mult=2,    # Multiply restart interval after each restart
        eta_min=1e-6 # Minimum learning rate
    )
    
    print("\nOptimization Setup:")
    print(f"  Loss Function: CrossEntropyLoss")
    print(f"  Optimizer: Adam")
    print(f"  LR Scheduler: CosineAnnealingWarmRestarts (T_0=10, T_mult=2)")
    
    # Training history
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # Track best model
    best_val_acc = 0.0
    best_epoch = 0
    
    # Training start time
    training_start = time.time()
    
    print("\n" + "="*70)
    print("Starting Training...")
    print("="*70)
    
    # Main training loop
    try:
        for epoch in range(num_epochs):
            # Training phase
            train_loss, train_acc = train_one_epoch(
                epoch, model, train_loader, criterion, optimizer, scheduler,
                train_losses, train_accuracies, device
            )
            
            # Validation phase
            val_loss, val_acc = validate_one_epoch(
                epoch, model, val_loader, criterion,
                val_losses, val_accuracies, device
            )
            
            # Summary for this epoch
            current_lr = scheduler.get_last_lr()[0]
            print(f"\n{'='*70}")
            print(f"Epoch [{epoch + 1}/{num_epochs}] Complete")
            print(f"{'='*70}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Check if this is the best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                print(f"\n🌟 New best validation accuracy! Saving model...")
                torch.save(model.state_dict(), 'gfnet_best_model.pth')
                print(f"✓ Best model saved: gfnet_best_model.pth")
            
            # Save checkpoint every 20 epochs
            if (epoch + 1) % 20 == 0:
                checkpoint_name = f'checkpoint_epoch_{epoch + 1}.pth'
                save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    train_losses, val_losses, train_accuracies, val_accuracies,
                    filename=checkpoint_name
                )
            
            # Check if target accuracy is reached
            if val_acc >= 80.0:
                print(f"\n🎯 Target accuracy of 80% reached!")
                print(f"   Validation Accuracy: {val_acc:.2f}%")
            
            print(f"{'='*70}\n")
            
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user!")
    except Exception as e:
        print(f"\n\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    # Training complete
    total_training_time = time.time() - training_start
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Total Training Time: {timedelta(seconds=int(total_training_time))}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"Final Validation Accuracy: {val_accuracies[-1]:.2f}%")
    
    # Save final model
    print("\nSaving final model...")
    torch.save(model.state_dict(), 'gfnet_final_model.pth')
    print("✓ Final model saved: gfnet_final_model.pth")
    
    # Save final checkpoint
    save_checkpoint(
        model, optimizer, scheduler, num_epochs - 1,
        train_losses, val_losses, train_accuracies, val_accuracies,
        filename='gfnet_final_checkpoint.pth'
    )

    # Plot and save metrics
    print("\nGenerating training plots...")
    plot_metrics(
        len(train_losses), train_losses, val_losses,
        train_accuracies, val_accuracies,
        save_path='Training_vs_validation.png'
    )

    # Final statistics
    print("\n" + "="*70)
    print("Final Statistics:")
    print("="*70)
    print(f"Training Loss:     {train_losses[-1]:.4f}")
    print(f"Training Accuracy: {train_accuracies[-1]:.2f}%")
    print(f"Validation Loss:   {val_losses[-1]:.4f}")
    print(f"Validation Accuracy: {val_accuracies[-1]:.2f}%")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")

    print("="*70 + "\n")


if __name__ == '__main__':
    main()
