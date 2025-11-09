# Day_013 | Project 2 | Graduate Admission Prediction using Keras Sequential API

## Overview

This project predicts a student’s chance of admission into graduate school using a simple **feedforward neural network**. The dataset typically includes features like:

* GRE Score (0–340)
* TOEFL Score (0–120)
* University Rating (1–5)
* Statement of Purpose (SOP) Strength (1–5)
* Letter of Recommendation (LOR) Strength (1–5)
* CGPA (0–10)
* Research Experience (0 or 1)

The **target variable** is the **Chance of Admission**, a continuous value between 0 and 1.

We use a **Sequential Keras model** with fully connected layers and ReLU activation, suitable for regression.

---

## Step 1: Import Libraries

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```

---

## Step 2: Load and Prepare Data

```python
# Load dataset
data = pd.read_csv("Admission_Predict.csv")  # CSV file containing the dataset

# Features and target
X = data[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']].values
y = data['Chance of Admit '].values

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

**Explanation**:

* Standardization ensures all features have mean 0 and variance 1, improving ANN training.
* `train_test_split` separates data for training and evaluation.

---

## Step 3: Define the ANN Model

```python
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Hidden layer 1
    Dense(32, activation='relu'),                                    # Hidden layer 2
    Dense(1)                                                         # Output layer (regression)
])
```

**Explanation**:

* `Dense(64)` and `Dense(32)` are fully connected layers.
* `relu` activation for hidden layers.
* **No activation** in the output layer for regression (predict continuous values).

---

## Step 4: Compile the Model

```python
model.compile(
    optimizer='adam',
    loss='mean_squared_error',  # MSE is standard for regression
    metrics=['mean_absolute_error']  # MAE provides interpretable error
)
```

**Explanation**:

* `MSE` (Mean Squared Error) penalizes larger errors.
* `MAE` (Mean Absolute Error) gives average absolute difference between predictions and true values.

---

## Step 5: Train the Model

```python
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)
```

**Explanation**:

* 100 epochs is usually sufficient for a small ANN.
* `validation_split=0.2` monitors overfitting.

---

## Step 6: Evaluate the Model

```python
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test MSE: {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")
```

**Explanation**:

* Evaluates performance on unseen test data.
* MAE gives an intuitive measure of average prediction error.

---

## Step 7: Make Predictions

```python
y_pred = model.predict(X_test[:5])
print("Predicted Chances of Admission:", y_pred.flatten())
print("Actual Chances of Admission:", y_test[:5])
```

**Explanation**:

* Predicts admission probability for new students.
* `.flatten()` converts output to 1D array for easy comparison.

---

## Optional: Visualize Training History

```python
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()
```

**Explanation**:

* Plots training and validation loss to check convergence and overfitting.

---

## Summary

* Used a **simple ANN** with 2 hidden layers for regression.
* Standardized features to improve learning stability.
* Achieved prediction of admission chances as a continuous value.
* Can be improved with **dropout, regularization, or hyperparameter tuning**.