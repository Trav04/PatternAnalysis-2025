# Alzheimer's Disease Classification Using GFNet

## Overview

This project implements a deep learning approach for binary classification of Alzheimer's disease (AD) using brain MRI scans from the ADNI dataset. The model distinguishes between Cognitive Normal (CN) and Alzheimer's Disease (AD) patients using Global Filter Networks (GFNet), a vision architecture that leverages Fourier domain processing for efficient global spatial feature extraction.

Alzheimer's disease is a progressive neurodegenerative disorder that affects memory and cognitive function. Early detection through neuroimaging is crucial for timely intervention. Traditional approaches rely on attention mechanisms and qualitative tests, but GFNet offers an alternative by performing computer vision analysis on brain scans from at risk patients. GFNet uses spatial mixing in the frequency domain using Fast Fourier Transforms (FFT) to perform this analysis. This approach is particularly effective for medical imaging where clear visual signs are strong indicators of the disease (such as brain atrophy patterns across different regions) and are prevelent in accurate diagnoses.

## Model Architecture

### How GFNet Works

GFNet replaces the self-attention mechanism with Fourier-based global filtering. The architecture processes brain MRI scans through the following pipeline visualised by a digram from Rao et al. (2023).

![GFNetDiagram](imgs/GFNetDiagram.png)

1. **Patch Embedding**: Divides input image into non-overlapping patches (16×16)
2. **Global Filter Layer**: 
   - Applies 2D FFT to transform patches to frequency domain
   - Multiplies frequency features with learnable global filters
   - Applies 2D inverse FFT to return to spatial domain
3. **Feed Forward Network (FFN)**: Layer normalization followed by MLP for feature refinement
4. **Global Average Pooling**: Aggregates patch features into a single representation
5. **Linear Classifier**: Maps pooled features to class predictions (CN/AD)


**Architecture Pipeline:**

Each GFNet block performs the following operations:
- Applies 2D FFT to convert spatial features to frequency domain
- Multiplies with learnable complex-valued filters
- Applies inverse FFT to return to spatial domain
- Feeds through MLP for channel mixing with residual connections

![Training Progress](imgs/TrainingGraphs.png)
*Figure: Training and validation loss/accuracy curves showing model convergence over 50 epochs*

## Dependencies

### Core Requirements

```
Python >= 3.8
PyTorch >= 1.12.0
torchvision >= 0.13.0
timm >= 0.6.12
nibabel >= 4.0.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
matplotlib >= 3.5.0
tqdm >= 4.62.0
```

### Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install timm nibabel numpy scikit-learn matplotlib tqdm
```

### Hardware Requirements

- **GPU**: NVIDIA GPU with at least 8GB VRAM (e.g., RTX 3070, V100)
- **RAM**: Minimum 16GB system memory

### Dependency Versions Used

The following specific versions were used in development and testing:

```
timm==1.0.20
matplotlib==3.10.7
tqdm==4.67.1
numpy==2.3.4
pillow==12.0.0
torch==2.9.0
torchvision==0.24.0
scikit-learn==1.7.2
```

## Dataset

### ADNI Brain MRI Data

The Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset contains preprocessed brain MRI scans with two classes:
- **CN (Cognitive Normal)**: Healthy control subjects
- **AD (Alzheimer's Disease)**: Patients diagnosed with AD

**Dataset Structure:**
```
/home/groups/comp3710/ADNI/
├── train/
│   ├── CN/  (Cognitive Normal)
│   └── AD/  (Alzheimer's Disease)
├── val/
│   ├── CN/
│   └── AD/
└── test/
    ├── CN/
    └── AD/
```

### Preprocessing Pipeline

The preprocessing steps applied to the raw MRI data include:

1. **Skull Stripping**: Removal of non-brain tissue to focus on brain structures
2. **Registration**: Spatial normalization to MNI152 template space for anatomical consistency
3. **Intensity Normalization**: Z-score normalization per scan: `(x - μ) / σ`
4. **Resampling**: Standardized to 224×224 pixel resolution
5. **Format**: Stored as NIfTI (.nii/.nii.gz) files with single-channel grayscale

**Data Augmentation** (applied during training only):
- Random horizontal flips (p=0.5)
- Random rotation (±10 degrees)
- Random affine transformations
- Intensity jittering

These augmentations help the model generalize better by simulating natural variations in brain anatomy and scan positioning.

### Dataset Splits

The data is split following standard practices in medical imaging:

- **Training Set**: 70% of data for model learning
- **Validation Set**: 15% for hyperparameter tuning and early stopping
- **Test Set**: 15% held out for final evaluation

**Split Justification**: The 70/15/15 split provides sufficient training examples while maintaining adequate validation and test sets for reliable performance estimation. Data is stratified by class to ensure balanced representation across splits. Patient-level splitting ensures no data leakage (scans from the same patient appear only in one split).

## Usage

### Project Structure

```
alzheimer_classification/
├── modules.py          # GFNet architecture implementation
├── dataset.py          # Data loading and preprocessing
├── train.py            # Training script with validation
├── predict.py          # Inference on new samples
├── README.md           # This file
└── requirements.txt    # Dependency list
```

### Training the Model

```bash
python train.py --data_path /home/groups/comp3710/ADNI \
                --model_variant base \
                --batch_size 32 \
                --epochs 50 \
                --learning_rate 1e-4 \
                --img_size 224
```

**Key Training Parameters:**
- `--model_variant`: Choose from `small`, `base`, or `large` (default: `base`)
- `--batch_size`: Batch size for training (default: 32)
- `--epochs`: Number of training epochs (default: 50)
- `--learning_rate`: Initial learning rate (default: 1e-4)
- `--img_size`: Input image resolution (default: 224)

**Training Strategy:**
- Optimizer: AdamW with weight decay 0.05
- Learning Rate Schedule: Cosine annealing with warmup
- Loss Function: Cross-Entropy Loss
- Early Stopping: Patience of 10 epochs based on validation loss

### Making Predictions

```bash
python predict.py --model_path best_model.pth \
                  --input_path /path/to/brain_scan.nii.gz \
                  --output_dir predictions/
```

## Example Inputs and Outputs

### Input Format

- **File Type**: NIfTI format (.nii or .nii.gz)
- **Image Dimensions**: 224 × 224 pixels (single 2D slice or preprocessed 3D volume)
- **Channels**: 1 (grayscale)
- **Intensity Range**: Normalized MRI intensity values

### Example Input



### Example Output


### Visualization Output

### Performance Metrics on Test Set

## Model Variants

Three GFNet variants are available with different capacity:

| Variant | Embed Dim | Depth | Params | Memory (GPU) |
|---------|-----------|-------|--------|--------------|
| Small   | 384       | 12    | ~25M   | ~6GB         |
| Base    | 768       | 12    | ~90M   | ~8GB         |
| Large   | 1024      | 18    | ~220M  | ~12GB        |

**Recommendation**: Start with `base` for optimal balance between accuracy and computational cost.

## File Descriptions

### modules.py

Contains the GFNet architecture implementation with the following components:

- `Mlp`: Multi-layer perceptron with GELU activation
- `GlobalFilter`: Fourier-based spatial mixing layer (core innovation)
- `Block`: GFNet transformer block combining GlobalFilter and MLP
- `PatchEmbed`: Converts images to patch embeddings
- `GFNet`: Main model class for classification
- Model variants: `gfnet_small()`, `gfnet_base()`, `gfnet_large()`

### dataset.py

Handles data loading and preprocessing:

- `ADNIDataset`: PyTorch Dataset class for ADNI brain scans
- NIfTI file loading utilities
- Preprocessing pipeline (normalization, augmentation)
- Train/val/test data loaders with stratification

### train.py

Complete training pipeline:

- Model initialization and configuration
- Training loop with validation
- Loss and metric computation
- Checkpoint saving (best model selection)
- Learning rate scheduling
- Plotting training curves

### predict.py

Inference script for new samples:

- Model loading from checkpoint
- Single sample or batch prediction
- Visualization of results
- Export predictions to CSV

## Results Discussion

The model achieves **80.2% test accuracy**, exceeding the required 80% threshold. The training curves (Figure 1) show:

- **Convergence**: Both training and validation losses decrease steadily over 50 epochs
- **Generalization**: Validation accuracy closely tracks training accuracy, indicating good generalization without significant overfitting
- **Stability**: Low variance in validation metrics suggests robust learning

**Key Observations**:

## Limitations and Future Work

**Current Limitations**:
- Binary classification only (CN vs AD); does not include MCI (Mild Cognitive Impairment)
- 2D slice-based analysis; 3D volumetric approach could capture more spatial information
- Limited to T1-weighted MRI; multimodal imaging (PET, DTI) could improve accuracy

**Future Improvements**:
- Implement multi-class classification (CN/MCI/AD)
- Add explainability methods (Grad-CAM, attention maps) for clinical interpretability
- Ensemble with other architectures (ConvNeXt, Swin Transformer)
- Cross-dataset validation on other AD datasets (OASIS, AIBL)
- Longitudinal analysis for disease progression prediction

---

# References
Y. Rao, W. Zhao, Z. Zhu, J. Zhou and J. Lu, "GFNet: Global Filter Networks for Visual Recognition," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 9, pp. 10960-10973, 1 Sept. 2023, doi: 10.1109/TPAMI.2023.3263824.


**Last Updated**: October 2025