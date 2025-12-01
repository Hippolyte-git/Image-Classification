# Image-Classification
Academic project exploring end-to-end ML engineering. Implements a CNN pipeline for CIFAR-10 classification with integrated MLOps telemetry using Weights &amp; Biases

# CIFAR-10 Image Classification & MLOps Pipeline

[![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Red?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![Weights & Biases](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white)](https://wandb.ai/site)

## 📋 Project Overview
This project was developed as part of the my course AI Engineering.

The primary objective was to engineer an end-to-end Machine Learning pipeline for Computer Vision. Moving beyond simple model training, this project focuses on **Machine Learning Operations (MLOps)** principles, implementing intelligent telemetry to track experiments, visualize loss curves, and manage model artifacts using the **Weights & Biases (W&B)** platform.

## 🎯 Objectives
*   **Architecture Comparison:** Transitioning from a naive Multi-Layer Perceptron (MLP) approach to a Convolutional Neural Network (CNN) to capture spatial features in image data.
*   **Data Engineering:** implementing data normalization and pipeline optimization strategies.
*   **Intelligent Telemetry:** Integrating `wandb` callbacks to stream real-time training metrics (loss, accuracy) and manage model checkpoints.
*   **Experiment Analysis:** Diagnosing model performance issues (underfitting/overfitting) based on telemetry data.

## 🛠️ Technologies & Tools
*   **Frameworks:** TensorFlow, Keras
*   **MLOps:** Weights & Biases (W&B)
*   **Data Processing:** NumPy, Matplotlib, Pandas
*   **Dataset:** CIFAR-10 (60,000 32x32 color images in 10 classes)

## 📊 Experiment Results & Analysis


**Experiment Configuration:**
*   **Model:** CNN (Conv2D + MaxPooling + GlobalAveragePooling)
*   **Data:** 100% of CIFAR-10 Training Data
*   **Epochs:** 10
*   **Learning Rate:** 0.005 (Adam Optimizer)

**Performance:**
*   **Test Accuracy:** 47.66%
*   **Test Loss:** 1.42

**Engineering Analysis:**
The telemetry from the latest run indicates that the model is currently **Underfitting**.
1.  **Metric Convergence:** Both training accuracy (~48%) and validation accuracy (~47%) plateaued quickly and remained close together.
2.  **Root Cause:** The Learning Rate of `0.005` was likely too aggressive for the Adam optimizer, preventing the model from converging to a lower loss. Additionally, the model architecture may require increased capacity (more layers) to capture the complexity of the full dataset.
3.  **Next Steps:** Future iterations will focus on hyperparameter tuning (reducing LR to `0.001`) and increasing model depth.



[![Weights & Biases](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white)](https://wandb.ai/site)
