"""
Training Script for GFNet-based Alzheimer's Disease Classification

This updated script refines the original GFNet training pipeline to address
training stagnation and improve stability. Enhancements include:
- Scheduler stepping after optimizer.step (required for OneCycleLR)
- Correct validation TTA using averaged logits for accurate loss computation
- Updated autocast context manager for compatibility with latest PyTorch
- Sanity checks on dataset integrity and label correctness
- Built-in overfit test to verify model learns on small subsets
- Scheduler step triggered only when an optimizer update occurs
- Simplified, robust default hyperparameters with CutMix disabled by default
- Clear, informative logging to aid debugging and ensure reliable training

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
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import time
from datetime import timedelta
import random
import torch.nn.functional as F
from torch import amp
import matplotlib.pyplot as plt

from modules import (
    create_enhanced_gfnet_small, 
    create_enhanced_gfnet_base,
    create_enhanced_gfnet_large
)
from dataset import build_data_pipeline


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with label smoothing to prevent overconfident predictions.
    Encourages the model to be less certain, improving generalization."""
    
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.size(-1) - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


class MixUp:
    """MixUp data augmentation that linearly interpolates between training examples.
    Creates synthetic samples by blending images and labels."""
    
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, images, labels, model, criterion):
        if self.alpha <= 0:
            return None
        
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(images.device)
        
        # Blend images
        mixed_images = lam * images + (1 - lam) * images[index]
        outputs = model(mixed_images)
        
        # Blend losses
        loss = lam * criterion(outputs, labels) + (1 - lam) * criterion(outputs, labels[index])
        return outputs, loss, labels, index, lam


class CutMix:
    """CutMix augmentation that replaces rectangular regions with patches from other images.
    More aggressive than MixUp, disabled by default (alpha=0.0)."""
    
    def __init__(self, alpha=0.0):
        self.alpha = alpha
    
    def __call__(self, images, labels, model, criterion):
        if self.alpha <= 0:
            return None
        
        batch_size, _, h, w = images.shape
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Determine cut region size
        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_ratio)
        cut_h = int(h * cut_ratio)
        
        # Random center point
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        # Bounding box coordinates
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Mix images by replacing region
        index = torch.randperm(batch_size).to(images.device)
        mixed_images = images.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda based on actual cut area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        outputs = model(mixed_images)
        loss = lam * criterion(outputs, labels) + (1 - lam) * criterion(outputs, labels[index])
        return outputs, loss, labels, index, lam


class EMA:
    """Exponential Moving Average of model parameters for more stable predictions.
    Maintains shadow weights that smooth out training fluctuations."""
    
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.register()
    
    def register(self):
        """Initialize shadow parameters with current model weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow parameters with exponential moving average."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Temporarily replace model weights with shadow weights for evaluation."""
        backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                backup[name] = param.data.clone()
                param.data = self.shadow[name]
        return backup
    
    def restore(self, backup):
        """Restore original model weights after shadow evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = backup[name]


def _sanity_checks(train_loader, val_loader):
    """Quick sanity check to verify label distribution in a sample batch.
    Helps catch data loading issues early in training."""
    import collections
    all_train_labels = []
    for _, labels in train_loader:
        all_train_labels.append(labels.cpu().numpy())
        break
    
    if len(all_train_labels) > 0:
        uniq, counts = np.unique(np.concatenate(all_train_labels), return_counts=True)
        print("Sanity check (sample batch labels):", dict(zip(uniq.tolist(), counts.tolist())))


def _overfit_small_batch_diagnostic(model, dataset_fn, device,
                                    batch_size=8, steps=800,
                                    lr=1e-2, weight_decay=0.0,
                                    use_label_smoothing=False):
    """Diagnostic test to verify model can overfit a small batch.
    Ensures model architecture and gradients are functioning correctly before full training."""
    
    # Build deterministic loaders without augmentation
    dls = dataset_fn(batch_size=batch_size, val_fraction=0.2, augment=False)
    if 'train' not in dls:
        train_loader = list(dls.values())[0]
    else:
        train_loader = dls['train']

    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)

    uniq, counts = torch.unique(labels, return_counts=True)
    print("Overfit diagnostic - batch label distribution:", dict(zip(uniq.cpu().tolist(), counts.cpu().tolist())))
    print(f"Image stats - min {images.min().item():.4f}, max {images.max().item():.4f}, mean {images.mean().item():.4f}, std {images.std().item():.4f}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if use_label_smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=0.05)
    else:
        criterion = nn.CrossEntropyLoss()

    # Use eval mode for deterministic forward (disables dropout)
    orig_training = model.training
    model.eval()
    torch.set_grad_enabled(True)

    success = False
    for i in range(1, steps + 1):
        opt.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Monitor gradient norms for debugging
        total_grad_norm_sq = 0.0
        grad_examples = []
        for name, p in model.named_parameters():
            if p.grad is not None:
                gnorm = p.grad.data.norm(2).item()
                total_grad_norm_sq += gnorm ** 2
                if len(grad_examples) < 3:
                    grad_examples.append((name, gnorm))
        total_grad_norm = total_grad_norm_sq ** 0.5

        opt.step()

        _, preds = outputs.max(1)
        acc = (preds == labels).float().mean().item() * 100.0

        if i == 1 or i % 20 == 0 or acc > 95.0:
            print(f"Step {i:4d}/{steps} | loss: {loss.item():.4f} | acc: {acc:.2f}% | grad_norm: {total_grad_norm:.4f}")
            for nm, g in grad_examples:
                print(f"   grad sample - {nm[:50]:50s}: {g:.4e}")

        if acc > 95.0:
            print(f"Overfit success at step {i}: acc={acc:.2f}% loss={loss.item():.4f}")
            success = True
            break

    # Restore original training mode
    if orig_training:
        model.train()
    else:
        model.eval()
    torch.set_grad_enabled(True)

    if not success:
        print(f"Overfit ended; final acc={acc:.2f}% loss={loss.item():.4f}")
    return success


def train_one_epoch_enhanced(epoch, model, train_loader, criterion, optimizer, 
                            scheduler, train_losses, train_accuracies, device,
                            mixup, cutmix, scaler, ema, use_amp=True, 
                            gradient_clip=1.0, accumulation_steps=2):
    """Train model for one epoch with gradient accumulation, mixed precision, and data augmentation.
    Includes MixUp/CutMix augmentation and EMA weight updates."""
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    print(f"\n{'='*70}")
    print(f"Epoch {epoch + 1} - Enhanced Training Phase")
    print(f"{'='*70}")
    
    batch_count = len(train_loader)
    epoch_start = time.time()
    optimizer.zero_grad()
    updates = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        batch_start = time.time()
        images = images.to(device)
        labels = labels.to(device)
        
        # Random selection of augmentation strategy
        use_mix = (mixup is not None) and (random.random() < 0.5)
        use_cut = (cutmix is not None) and (random.random() < 0.2)

        with amp.autocast('cuda', enabled=use_amp):
            # Apply CutMix after warmup period
            if use_cut and (epoch > 5) and (cutmix is not None):
                out = cutmix(images, labels, model, criterion)
                if out is None:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    orig_labels = labels
                    index = None
                    lam = 1.0
                else:
                    outputs, loss, orig_labels, index, lam = out

            # Apply MixUp after warmup period
            elif use_mix and (epoch > 3) and (mixup is not None):
                out = mixup(images, labels, model, criterion)
                if out is None:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    orig_labels = labels
                    index = None
                    lam = 1.0
                else:
                    outputs, loss, orig_labels, index, lam = out

            else:
                # Standard forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                orig_labels = labels
                index = None
                lam = 1.0

        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Perform optimizer step when accumulation threshold reached
        if (batch_idx + 1) % accumulation_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            updates += 1
            
            # Step scheduler after optimizer update (required for OneCycleLR)
            if scheduler is not None:
                scheduler.step()
            
            ema.update()
        
        # Track training statistics
        running_loss += loss.item() * accumulation_steps
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        
        # Adjust accuracy calculation for mixed samples
        if lam == 1.0:
            correct += (predicted == orig_labels).sum().item()
        else:
            correct += lam * (predicted == orig_labels).sum().item()
        
        batch_time = time.time() - batch_start
        current_acc = 100.0 * correct / total
        current_lr = scheduler.get_last_lr()[0] if scheduler is not None else 0.0
        
        # Periodic logging
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == batch_count:
            print(f"  Batch [{batch_idx + 1:4d}/{batch_count:4d}] | Loss: {loss.item() * accumulation_steps:.4f} | "
                  f"Acc: {current_acc:.2f}% | LR: {current_lr:.6f} | Time: {batch_time:.2f}s")
    
    # Final step if batches don't divide evenly by accumulation_steps
    if batch_count % accumulation_steps != 0:
        if use_amp:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()
        ema.update()
    
    epoch_loss = running_loss / batch_count
    epoch_acc = 100.0 * correct / total if total > 0 else 0.0
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    
    epoch_time = time.time() - epoch_start
    print(f"\n{'─'*70}")
    print(f"Training Summary:")
    print(f"  Average Loss: {epoch_loss:.4f}")
    print(f"  Accuracy: {epoch_acc:.2f}%")
    print(f"  Epoch Time: {timedelta(seconds=int(epoch_time))}")
    print(f"{'─'*70}")
    
    return epoch_loss, epoch_acc


def validate_with_tta(model, val_loader, criterion, device, tta_transforms=None):
    """Validate model with Test-Time Augmentation for improved robustness.
    Averages predictions across multiple augmentations of each image."""
    
    model.eval()
    if tta_transforms is None:
        # Default: original and horizontal flip
        tta_transforms = [lambda x: x, lambda x: torch.flip(x, dims=[3])]
    
    all_preds = []
    all_labels = []
    running_loss = 0.0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Collect logits from each augmentation
            batch_logits = []
            for transform in tta_transforms:
                aug_images = transform(images)
                with amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    outputs = model(aug_images)
                    batch_logits.append(outputs)
            
            # Average logits before applying softmax (proper ensembling)
            avg_logits = torch.stack(batch_logits).mean(dim=0)
            loss = criterion(avg_logits, labels)
            running_loss += loss.item()
            
            probs = F.softmax(avg_logits, dim=1)
            all_preds.append(probs.cpu())
            all_labels.append(labels.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    _, predicted = torch.max(all_preds, 1)
    correct = (predicted == all_labels).sum().item()
    total = all_labels.size(0)
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    avg_loss = running_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
    
    return avg_loss, accuracy, predicted, all_labels


def main():
    """Main training function that orchestrates the complete training pipeline.
    Includes data loading, model initialization, training loop, and result visualization."""
    
    print("Starting enhanced training (improved) ...")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters (tuned for stability and performance)
    batch_size = 32
    base_lr = 3e-4
    num_epochs = 80
    weight_decay = 1e-4
    label_smoothing = 0.00
    mixup_alpha = 0.0
    cutmix_alpha = 0.0
    gradient_clip = 1.0
    accumulation_steps = 2
    ema_decay = 0.999
    
    print("Loading data...")
    data_loaders = build_data_pipeline(batch_size=batch_size, val_fraction=0.2, augment=True)
    train_loader = data_loaders['train']
    val_loader = data_loaders['val']
    print(f"Train dataset size (subset len): {len(train_loader.dataset)}")
    print(f"Val dataset size (subset len): {len(val_loader.dataset)}")
    
    _sanity_checks(train_loader, val_loader)
    
    print("Initializing model...")
    model = create_enhanced_gfnet_base(num_classes=2, img_size=224, in_chans=3)
    model = model.to(device)
    
    ema = EMA(model, decay=ema_decay)
    criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    
    steps_per_epoch = max(1, len(train_loader) // accumulation_steps)
    scheduler = OneCycleLR(optimizer, max_lr=3e-3, epochs=num_epochs, 
                          steps_per_epoch=max(1, len(train_loader)//accumulation_steps), 
                          pct_start=0.15, div_factor=10, final_div_factor=100)
    
    use_amp = torch.cuda.is_available()
    scaler = amp.GradScaler(enabled=use_amp)
    mixup = MixUp(alpha=mixup_alpha)
    cutmix = CutMix(alpha=cutmix_alpha)
    
    # Display model information
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {num_params:,}, trainable: {num_trainable:,}")

    # Sample label distribution check
    sample_labels = []
    for _, lbl in train_loader:
        sample_labels.append(lbl.cpu().numpy())
        if len(sample_labels) >= 5: break
    sample_labels = np.concatenate(sample_labels)
    print("Sample labels unique:", np.unique(sample_labels), "counts:", np.bincount(sample_labels.astype(int)))

    print("Running quick overfit test on a single batch to ensure model can learn...")
    overfit_ok = _overfit_small_batch_diagnostic(
        model,
        dataset_fn=build_data_pipeline,
        device=device,
        batch_size=8,
        steps=400,
        lr=5e-3,
        weight_decay=0.0,
        use_label_smoothing=False
    )
    if not overfit_ok:
        print("Warning: model failed to overfit small batch. Inspect data, loss, model and try again.")
    
    # Training loop with metrics tracking
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_acc = 0.0
    patience = 20
    patience_counter = 0
    
    try:
        for epoch in range(num_epochs):
            train_loss, train_acc = train_one_epoch_enhanced(
                epoch, model, train_loader, criterion, optimizer,
                scheduler, train_losses, train_accuracies,
                device, mixup, cutmix, scaler, ema,
                use_amp, gradient_clip, accumulation_steps
            )
            
            # Validation with standard weights
            val_loss, val_acc, _, _ = validate_with_tta(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            # Validation with EMA weights
            backup = ema.apply_shadow()
            ema_loss, ema_acc, _, _ = validate_with_tta(model, val_loader, criterion, device)
            ema.restore(backup)
            
            print(f"Epoch {epoch+1}/{num_epochs} | Train acc: {train_acc:.2f}% | Val acc: {val_acc:.2f}% | EMA Val: {ema_acc:.2f}%")
            
            # Save best model based on validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), 'gfnet_best_model.pth')

    except KeyboardInterrupt:
        print("Interrupted")
    except Exception as e:
        import traceback
        traceback.print_exc()
    
    print("Training complete. Best val acc:", best_val_acc)
    
    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs_range = range(1, len(train_losses) + 1)
    
    # Loss plot
    ax1.plot(epochs_range, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs_range, val_losses, 'orange', label='Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss vs Epochs')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs_range, train_accuracies, 'b-', label='Train Accuracy')
    ax2.plot(epochs_range, val_accuracies, 'orange', label='Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy vs Epochs')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_plots.png', dpi=300, bbox_inches='tight')
    print("Training plots saved as 'training_plots.png'")
    plt.show()


if __name__ == '__main__':
    main()