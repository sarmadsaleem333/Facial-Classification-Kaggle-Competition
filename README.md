# Facial Classification Deep Learning Models

This repository contains the code and methodology for training and evaluating deep learning models on a facial classification task using different architectures. The models evaluated include **ResNet-50**, **MobileNetV2**, and **InceptionResNetV1**, with detailed results for each.

## 📝 Project Overview

This project focuses on building and evaluating deep learning models for the task of **facial classification**. Several models were tested and compared to determine the best architecture for a large-scale classification task with **7,000 classes**. We used publicly available pre-trained models and fine-tuned them on a custom dataset.

## ⚡ Models Used

### 1. **ResNet-50**
- **Description**: ResNet-50 is a 50-layer deep convolutional neural network (CNN) that utilizes residual connections to address the vanishing gradient problem. It is known for its deep architecture, which helps in learning rich features from complex data.
- **Training Details**:
  - Optimizer: Adam (`β1=0.9`, `β2=0.999`)
  - Batch Size: 64
  - Learning Rate: `1e-3` (reduced on plateau)
  - Epochs: 17 (3 frozen + 14 fine-tuned)
  - Dropout: 0.5
  - Loss Function: Categorical Cross-Entropy with Label Smoothing (`ε=0.1`)

### 2. **MobileNetV2**
- **Description**: MobileNetV2 is a lightweight architecture designed for efficient computation on mobile and embedded devices. It uses depthwise separable convolutions and inverted residuals with linear bottlenecks, making it ideal for environments where computational resources are limited.
- **Training Details**:
  - Optimizer: AdamW (`weight_decay=1e-4`)
  - Batch Size: 64
  - Learning Rate: `5e-4`
  - Epochs: 10 (2 frozen + 8 fine-tuned)
  - Dropout: 0.5
  - Scheduler: ReduceLROnPlateau (patience=2)

### 3. **InceptionResNetV1**
- **Description**: InceptionResNetV1 combines the strengths of Inception modules and residual learning. It was pre-trained on the VGGFace2 dataset, which made it particularly effective for face classification tasks. In this experiment, we froze the initial layers to avoid overfitting on a small dataset and fine-tuned the last 15 layers.
- **Training Details**:
  - Optimizer: Adam (`β1=0.9`, `β2=0.999`)
  - Batch Size: 64
  - Learning Rate: `1e-3` → `5e-4`
  - Epochs: 10 (3 frozen + 7 unfrozen)
  - Dropout: 0.5
  - Label Smoothing: 0.1
  - Scheduler: ReduceLROnPlateau (patience=2)

## 📊 Evaluation Results

After training all three models, their performance was evaluated on validation and test sets, and further validated on the Kaggle leaderboard for the competition.

### Model Performance Summary

| Model             | Validation Accuracy (%) | Test Accuracy (%) | Kaggle Score (%) |
|-------------------|-------------------------|-------------------|------------------|
| ResNet-50         | 51.76                   | 51.91             | 50.00            |
| MobileNetV2       | 57.24                   | 56.74             | 55.00            |
| **InceptionResNetV1** | **80.39**           | **79.02**         | **79.00**        |

### Best Performing Model

**InceptionResNetV1** emerged as the top performer across all evaluation stages, showing the highest accuracy across validation, test, and Kaggle leaderboard. It provided the best trade-off between model complexity and accuracy.

