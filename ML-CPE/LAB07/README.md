#  Coffee Bean Roast Level Classification (CNN Pipeline)

An end-to-end Computer Vision project that classifies coffee bean roast levels into four distinct categories using a Convolutional Neural Network (CNN) built with TensorFlow/Keras.

---

##  Table of Contents
- [Overview](#-overview)
- [Dataset Architecture](#-dataset-architecture)
- [Project Directory](#-project-directory)
- [Key Features](#-key-features)
- [Performance & Results](#-performance--results)
- [Getting Started](#-getting-started)
- [Model Evaluation & Outputs](#-model-evaluation--outputs)

---

##  Overview

This project provides an automated deep learning pipeline designed to identify coffee roast levels: **Dark**, **Green**, **Light**, and **Medium**. 

To tackle severe overfitting common in small-scale image recognition tasks, the architecture incorporates real-time Data Augmentation (random flips, rotation, zoom, brightness adjustments) alongside Dropout and Batch Normalization layers.

- **Input Resolution:** 100 x 100 pixels (RGB)
- **Framework:** TensorFlow 2.x / Keras (Python 3.11)
- **Final Test Accuracy:** **99.58%**

---

##  Dataset Architecture

The project expects dataset images to be structured into class-specific subdirectories under `dataset/train/`: [text](https://www.kaggle.com/datasets/gpiosenka/coffee-bean-dataset-resized-224-x-224)


```text
dataset/
└── train/
    ├── Dark/       # [300 images]
    ├── Green/      # [300 images]
    ├── Light/      # [300 images]
    └── Medium/     # [300 images]


 ##  CNN Model Architecture

The neural network utilizes a stacked convolutional architecture optimized for fine-grained image recognition:

1. Input Layer: Accepts (100, 100, 3) RGB tensor inputs.
2. Data Augmentation Block: On-the-fly transformations applied during training.
3. Feature Extraction ConvBlocks (x3):
   - Conv2D (32, 64, and 128 filters with 3x3 kernels & ReLU activation)
   - BatchNormalization for stabilizing learning rates and gradient flow
   - MaxPooling2D (2x2 spatial downsampling)
4. Classification Head:
   - GlobalAveragePooling2D to reduce feature dimensions and control parameters
   - Dense hidden layer with 128 units
   - Dropout (rate = 0.5) to prevent co-adaptation
   - Dense output layer with Softmax activation across 4 roast classes

---
## Performance & Results

Evaluating the trained network on an independent test partition (20% of total dataset, N=240) yielded near-perfect classification performance:

- Test Accuracy: 99.58%
- Macro Average Precision: 1.00
- Macro Average Recall: 1.00
- Macro Average F1-Score: 1.00

Detailed Classification Breakdown:
- Dark   : Precision 1.00 | Recall 0.98 | F1-Score 0.99 | Support 60
- Green  : Precision 1.00 | Recall 1.00 | F1-Score 1.00 | Support 60
- Light  : Precision 1.00 | Recall 1.00 | F1-Score 1.00 | Support 60
- Medium : Precision 0.98 | Recall 1.00 | F1-Score 0.99 | Support 60

---