# Vehicle Damage Detection Project

Deployed Application link: https://vehicle-damage-detection-by-abhijeet-singh-pawar-j6muvtcujycob.streamlit.app/

## Project Overview

This repository contains a deep learning-based project for detecting and classifying vehicle damage using image data. The model categorizes vehicle images into six classes:  
- **F_Breakage** (Front Breakage)  
- **F_Crushed** (Front Crushed)  
- **F_Normal** (Front Normal)  
- **R_Breakage** (Rear Breakage)  
- **R_Crushed** (Rear Crushed)  
- **R_Normal** (Rear Normal)  

The project leverages a pre-trained ResNet50 model fine-tuned on a dataset of 2,300 images. It includes data loading, model training, hyperparameter tuning with Optuna, evaluation via confusion matrices, and model saving.  

Key achievements:  
- Achieved ~80% validation accuracy after tuning.  
- Handles data augmentation for robustness (e.g., flips, rotations, color jitter).  

Current as of November 30, 2025.

## Repository Contents

- `damage_prediction (1).ipynb`: Main notebook for data loading, model training, validation, confusion matrix visualization, and model saving.  
- `hyperparameter_tunning (1).ipynb`: Notebook focused on hyperparameter optimization using Optuna (tuning learning rate and dropout rate).  
- `README.md`: This file.  

**Note**: The dataset (located in `./dataset`) is not included in this repository due to size constraints. It assumes a folder structure with subfolders for each class.

## Key Components

### 1. Data Preparation  
- Dataset: 2,300 images split into 75% training (1,725) and 25% validation (575).  
- Transformations: Random horizontal flips, rotations (10°), color jitter, resizing to 224x224, normalization (ImageNet stats).  
- Loader: PyTorch DataLoader with batch size 32 and shuffling.

### 2. Model Architecture  
- Base: Pre-trained ResNet50 from torchvision.  
- Modifications: Freeze all layers except the final layer4 and fully connected (FC) layer.  
- FC Layer: Dropout + Linear layer for 6-class output.  
- Trained on GPU (CUDA) if available.

### 3. Training & Evaluation  
- Optimizer: Adam (with tuned learning rate).  
- Loss: Cross-Entropy.  
- Epochs: 3 (for tuning) to 10+ (full training).  
- Metrics: Accuracy, confusion matrix (visualized with Matplotlib and scikit-learn).  

### 4. Hyperparameter Tuning  
- Tool: Optuna.  
- Parameters Tuned: Learning rate (1e-5 to 1e-2), Dropout rate (0.2 to 0.7).  
- Trials: 20, with pruning for efficiency.  
- Best Params (example from run): lr ≈ 0.0004, dropout ≈ 0.66 (results may vary due to randomness).

### 5. Results  
- Validation Accuracy: Up to 80% post-tuning.  
- Confusion Matrix: Highlights common misclassifications (e.g., between breakage and crushed categories).  
- Model saved as `saved_model.pth` for inference.

## Dependencies

Install via pip:  
```bash
pip install torch torchvision optuna matplotlib numpy scikit-learn
