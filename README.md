<----------------------------------------Neural Network from Scratch with Numpy------------------------------------------------------->

PROJECT OVERVIEW
Handwritten Digit Recognition using feedforward Neural Network built entirely from scratch with NumPy only.
Goal: Classify MNIST digits (0-9) achieving 97.99% test accuracy on 10K test set.


DATA COLLECTION
Dataset: MNIST Handwritten Digits
- Training: 60,000 images (28×28 pixels, grayscale)
- Test: 10,000 images (28×28 pixels, grayscale)
- Labels: 0-9 (10 classes)
Source: MNIST dataset via 'mnist' library


TECHNOLOGIES USED
Core: NumPy (matrix operations)
Data: mnist, sklearn (OneHotEncoder)
Visualization: matplotlib
Testing: unittest (gradient checking)


DATA PREPROCESSING TECHNIQUES
1. Pixel Normalization: [0-255] → [0,1]
   X_train = np.array(X_train) / 255.0

2. Label Encoding: One-hot encoding
   y_train: [5] → [0,0,0,0,0,1,0,0,0,0]

3. Train/Validation Split: 90/10 ratio
   54K training → 6K validation

4. Batch Processing: Mini-batches of size 64



MODEL ARCHITECTURE
Input Layer (784 neurons - 28×28 flattened)
↓
Hidden Layer 1 (128 neurons, ReLU activation)
↓
Hidden Layer 2 (64 neurons, ReLU activation)
↓
Output Layer (10 neurons, Softmax activation)

Total Parameters: ~50K weights + biases

CODE IMPLEMENTATION DETAILS
Xavier (He) Weight Initialization: sqrt(2/input_size)
Forward Pass: Linear → ReLU → Softmax pipeline
Backward Pass: Complete backpropagation with chain rule
Loss Function: Categorical Cross-Entropy
Optimizer: Mini-batch SGD (learning_rate=0.1)
Regularization: Early stopping (patience=5 epochs)
Validation: Numerical gradient checking via unit tests



TRAINING RESULTS ACHIEVED
Training Duration: 21 epochs (early stopping triggered)



Final Performance Metrics:
├── Training Loss: 0.0034 (99.99% accuracy)
├── Validation Loss: 0.0761 (98.15% accuracy)
└── Test Accuracy: 97.99% (10,000 test samples)

Visualization Generated: traininghistory.png

USAGE INSTRUCTIONS
1. Install dependencies:
   pip install mnist sklearn matplotlib

2. Run training:
   jupyter notebook NeuralNetwork-from-Scratch-with-Numpy-ML-Task.ipynb

PROJECT FILES
├── NeuralNetwork-from-Scratch-with-Numpy-ML-Task.ipynb  (Main code)
├── traininghistory.png                                 (Results plot)
└── Neural_Network_README.txt                           (This file)

CONCLUSION & KEY LEARNINGS
Achieved production-quality 97.99% accuracy with pure NumPy
Complete backpropagation implementation with gradient verification
Proper Xavier initialization ensures fast convergence (21 epochs)
Early stopping prevents overfitting effectively
Mini-batch SGD with validation monitoring works reliably
Unit tests confirm mathematical correctness


Pure NumPy Implementation | 97.99% Test Accuracy | Fully Tested & Documented
