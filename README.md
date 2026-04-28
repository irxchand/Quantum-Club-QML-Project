# Classical vs Quantum Models for Non-Linear Classification

## Overview

This project explores non-linear classification using both classical neural networks and Parameterized Quantum Circuits (PQC).

It demonstrates how quantum-inspired models can approximate non-linear decision boundaries and compares them against standard neural network approaches.

---

## Motivation

Non-linear problems such as XOR cannot be solved using simple linear models. This project investigates:

- How classical neural networks learn non-linear patterns
- How quantum circuits can achieve similar behavior using state transformations
- Differences in training dynamics and efficiency

---

## Project Components

### 1. Classical Neural Network (Baseline)

- 2-layer neural network using PyTorch
- Activation: Tanh + Sigmoid
- Loss: Binary Cross Entropy
- Optimizers compared:
  - SGD
  - Adam

 Key Output:
- Epoch convergence comparison
- Training time differences

---

### 2. PQC-based XOR Classifier

- Built using Qiskit
- Uses:
  - RY rotations for encoding
  - CX gates for entanglement
- Measurement via probability extraction

  Observations:
- Can learn XOR (non-linear function)
- Uses fewer parameters than classical NN

---

### 3. PQC for 2D Non-Linear Classification

- Data re-uploading architecture
- Multiple parameterized layers
- RY + RZ rotations
- Entanglement between qubits

Includes:
- Training using numerical gradient descent
- Evaluation on noisy test samples
- Confusion matrix visualization

---

### 4. Generalized PQC Trainer

- Supports:
  - arbitrary number of features
  - custom datasets
- Implements:
  - circuit generation
  - probability-based prediction
  - training loop with finite difference gradients

---

## System Flow


Input Data → Encoding (Rotations) → PQC Layers → Measurement → Probability Output → Classification


---

## Key Concepts Demonstrated

- Non-linear decision boundaries (XOR)
- Quantum state encoding
- Entanglement as feature interaction
- Parameterized circuits as ML models
- Gradient-based optimization (finite difference)

---

## How to Run

### 1. Install dependencies


pip install torch qiskit numpy matplotlib seaborn scikit-learn


---

### 2. Run Classical Model


python "Non Linear (NN).py"


---

### 3. Run PQC Demo


python "qml demo (pqc).py"


---

### 4. Run Generalized PQC Trainer


python "qml gen.py"


---

## Results Summary

- Classical NN converges faster (especially with Adam)
- PQC models successfully learn non-linear mappings
- PQCs require more computational effort (finite difference gradients)
- Demonstrates conceptual viability of quantum ML

---

## Limitations

- Uses statevector simulation (not real quantum hardware)
- Finite-difference gradients are inefficient
- Small datasets only
- No hybrid classical-quantum optimization

---

## Future Improvements

- Parameter shift rule for gradients
- Hybrid quantum-classical models
- Integration with real quantum backends
- Scaling to larger datasets
- Variational quantum circuits

---

## Author

- Ishaan Chand
- Dharmit Shah
- Atharva Ambilwade
- Yashasv Khullar

---
