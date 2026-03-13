
# 🤖 Tabular Data Synthesis with GANs & cGANs

> **Generating high-fidelity synthetic tabular data using Deep Generative Models on the Adult Census Income dataset.**

## 📌 Project Overview

This project explores the capabilities of Generative Adversarial Networks (GANs) to synthesize complex tabular data. Using the **Adult Census Income dataset**, the goal was to create synthetic records that preserve the statistical properties and predictive utility of the original data—specifically for predicting whether an individual's income exceeds $50K/year.

## 🧪 Key Objectives

* **Fidelity:** Can the generator produce samples that are indistinguishable from real data?
* **Efficacy:** Can a model trained on synthetic data perform as well as one trained on real data? (Targeting a ratio close to 1.0).
* **Stability:** Comparing standard GAN training vs. Conditional GAN (cGAN) stability.
* **Rigorous Testing:** Results are averaged over **3 different random seeds** with an **80/20 train-test split**.

## 🏗️ Model Architectures

The project implements two custom PyTorch architectures optimized via grid search:

### 1. Standard GAN

* **Latent Space:** 64-dimensional random noise.
* **Hidden Layers:** Fully connected layers `[128, 128]`.
* **Architecture:** Uses `ReLU` for the Generator and `LeakyReLU` for the Discriminator, stabilized with `BatchNorm1d`.

### 2. Conditional GAN (cGAN)

* **Latent Space:** 32-dimensional noise + Class Labels.
* **Hidden Layers:** Increased capacity with `[256, 256]` units.
* **Implementation:** Incorporates class labels into the input layer to allow for targeted synthesis of specific income groups.

## 📈 Evaluation & Results

We used the **Random Forest** classifier as our primary evaluation tool under a "Train on Synthetic, Test on Real" (TSTR) framework.

### Statistical Fidelity (Detection)

* **Method:** A Random Forest was trained to classify "Real vs. Synthetic" data.
* **Result:** The models achieved high detection AUCs (close to **1.00**), indicating that a powerful classifier can still distinguish synthetic records from real ones, though cGAN significantly improved correlation preservation.

### Predictive Efficacy (Utility)

* **Method:** Calculated the ratio: `Synthetic Data AUC / Original Data AUC`.
* **Standard GAN:** Achieved a ratio of **0.674**. While it captured basic trends, it struggled with complex non-linear feature relationships.
* **Conditional GAN:** Achieved a ratio of **0.939**. This demonstrates that the cGAN-generated data is a highly effective substitute for real data in downstream machine learning tasks.

## 🛠️ Technical Implementation

* **Preprocessing:** Handled missing values (`?`), performed **One-Hot Encoding** for categorical features, and used **StandardScaler** for continuous variables (Age, fnlwgt, capital-gain, etc.).
* **Loss Function:** Utilized `BCEWithLogitsLoss` for numerical stability, which implicitly handles the Sigmoid activation.
* **Hyperparameter Optimization:** A grid search was performed across learning rates (1e-4 to 1e-6), dropout rates (0.0 to 0.3), and batch sizes.

