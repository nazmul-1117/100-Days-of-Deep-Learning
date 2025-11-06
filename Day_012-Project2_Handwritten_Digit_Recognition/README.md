# Day_012 | Handwritten Digit Recognition using ANN, Keras| MNIST Dataset

## Overview

This project implements a neural network to classify handwritten digits (0–9) from the **MNIST dataset** using Keras’s **Sequential API**. MNIST is a standard dataset of 70,000 grayscale images of size 28×28 pixels.

The model architecture is simple but effective:

1. **Flatten Layer**: Converts 2D images (28×28) into 1D vectors (784 features).
2. **Dense Layer**: Fully connected layer with 128 neurons and ReLU activation.
3. **Output Layer**: 10 neurons (one per class) with softmax activation for probability prediction.

---

## Step 1: Import Libraries

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
```

---

## Step 2: Load and Prepare Data

```python
# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize images to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encode labels (0–9)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
```

**Explanation**:

* Normalization helps the network train faster and improves accuracy.
* One-hot encoding converts labels into vectors like `[0, 0, 1, 0, ...]` for 10 classes.

---

## Step 3: Define the Model

```python
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten 28x28 images to 784-element vector
    Dense(128, activation='relu'),  # Hidden layer with 128 neurons
    Dense(10, activation='softmax') # Output layer for 10 classes
])
```

**Explanation**:

* `Flatten`: Converts 2D input to 1D vector.
* `Dense`: Fully connected layer.
* `relu`: Rectified Linear Unit, standard for hidden layers.
* `softmax`: Converts outputs to probabilities for classification.

---

## Step 4: Compile the Model

```python
model.compile(
    optimizer='adam',               # Adaptive optimizer
    loss='categorical_crossentropy', # Suitable for multi-class classification
    metrics=['accuracy']             # Track accuracy during training
)
```

**Explanation**:

* `Adam` optimizer adjusts learning rate automatically.
* `Categorical crossentropy` is standard for multi-class problems.

---

## Step 5: Train the Model

```python
history = model.fit(
    x_train, y_train,
    epochs=10,           # Number of training iterations over dataset
    batch_size=32,       # Number of samples per gradient update
    validation_split=0.2 # Use 20% of training data for validation
)
```

**Explanation**:

* `validation_split`: Helps monitor overfitting.
* `history` stores training metrics (loss, accuracy) for plotting later.

---

## Step 6: Evaluate the Model

```python
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
```

**Explanation**:

* Evaluate on unseen test data to check generalization.

---

## Step 7: Make Predictions

```python
import numpy as np

# Predict first 5 test images
predictions = model.predict(x_test[:5])
predicted_classes = np.argmax(predictions, axis=1)
print("Predicted labels:", predicted_classes)
```

**Explanation**:

* `np.argmax` converts probability vectors to class labels.

---

## Optional: Visualize Training History

```python
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

---

## Summary

* The **Sequential API** allows quick prototyping.
* Simple **MLP (fully connected network)** can achieve ~97–98% accuracy on MNIST.
* Normalization and proper one-hot encoding are critical.
* This model can be further improved with **Convolutional layers** or **Dropout** for regularization.

## Images
![image](assets/MNIST_dataset_example.png)