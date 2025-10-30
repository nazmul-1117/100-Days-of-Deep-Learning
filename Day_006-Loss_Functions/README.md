# Day 006 | Perceptron Loss and Activation

## 1. Perceptron's Traditional "Loss" (Update Rule)

The original perceptron, developed by Frank Rosenblatt, doesn't rely on a loss function for optimization; it relies on a discrete error signal:

* **Error Calculation:** $\text{Error} = \text{Target Output} (y) - \text{Predicted Output} (\hat{y})$
* **Goal:** The algorithm only updates the weights and bias when the prediction is **wrong** (i.e., when $\text{Error} \neq 0$). The weights remain unchanged when the classification is correct.
* **Use Case:** This method is guaranteed to converge to a perfect solution **if and only if** the data is **linearly separable** (meaning a single straight line or plane can perfectly separate the two classes).

### 2. Perceptron's Activation Function

* **Function:** **Heaviside Step Function** (or Sign Function for $\{-1, 1\}$ output)
* **Formula (Heaviside):**
    $$\hat{y} = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \le 0 \end{cases}$$
    where $z = \mathbf{w} \cdot \mathbf{x} + b$
* **Use Case:** Binary classification with a hard-cut threshold.

***

## Another Loss Function: Binary Cross-Entropy (Log Loss)

**Binary Cross-Entropy (BCE)** is the most common loss function used for binary classification in modern neural networks.

| Feature | Description |
| :--- | :--- |
| **Formula** | $L(\hat{y}, y) = - [y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})]$ |
| **Activation Pairing** | **Sigmoid** activation function. |
| **Use Case** | Binary classification problems where the output should be interpreted as a **probability** between 0 and 1. |
| **Why it's used** | It penalizes false predictions severely, especially when the predicted probability is confident but wrong. It is **differentiable**, which is necessary for the **Backpropagation** algorithm to work. |

***

## Another Activation Function: ReLU

The **Rectified Linear Unit (ReLU)** is the most widely used activation function in the hidden layers of modern deep learning models.

| Feature | Description |
| :--- | :--- |
| **Formula** | $f(z) = \max(0, z)$ |
| **Loss Pairing** | Typically **Cross-Entropy** (for classification) or **Mean Squared Error** (for regression). |
| **Use Case** | Used in the **hidden layers** of multi-layer perceptrons (deep neural networks) for both classification and regression. |
| **Why it's used** | It introduces the necessary **non-linearity** that allows neural networks to learn complex functions and patterns (solving non-linear problems like XOR). It is computationally efficient and helps solve the vanishing gradient problem common with older functions like Sigmoid. |
| **Formula** |  |


## Some more Loss - Activation Function
| Loss Function | Activation Function | Model / Use Case | Key Differentiator |
| :--- | :--- | :--- | :--- |
| **Perceptron Loss (Error-Driven Update)** | **Step Function** | **Perceptron** (Binary Classifier) | Only updates on **misclassified** points; non-differentiable. |
| **Hinge Loss** | **Sign/Identity** (Output $\in \{-1, 1\}$) | **Support Vector Machine (SVM)** / Maximum-Margin Classifier | Minimizes misclassification while maximizing the margin. |
| **Binary Cross-Entropy (Log Loss)** | **Sigmoid** ($f(z) = \frac{1}{1 + e^{-z}}$) | **Binary Classification** (e.g., Logistic Regression, Output Layer of Neural Network) | Output is a probability (0 to 1); used for **differentiable** learning (Backpropagation). |
| **Categorical Cross-Entropy** | **Softmax** ($\frac{e^{z_i}}{\sum e^{z_j}}$) | **Multi-Class Classification** (Output Layer of Neural Network) | Output is a probability distribution across $>2$ classes (sums to 1). |
| **Mean Squared Error (MSE)** | **Identity** ($f(z)=z$) / **ReLU** ($f(z)=\max(0, z)$) | **Regression** (e.g., Linear Regression, Prediction Layer of Neural Network) | Measures the square of the difference between predicted and actual values. |
| **L1 Loss (Mean Absolute Error)** | **Identity** ($f(z)=z$) | **Regression** (More Robust to Outliers than MSE) | Measures the absolute difference; results in a piecewise linear loss function. |

### Key Clarifications:

1.  **Perceptron:** The original perceptron does not use a differentiable loss function in the modern sense. Its "loss" is simply the binary error signal ($y - \hat{y}$), which drives the update.
2.  **Sigmoid/Softmax:** These are specifically used in the **output layer** for classification because they produce values that can be interpreted as probabilities.
3.  **ReLU (Rectified Linear Unit):** While often paired with **Cross-Entropy** or **MSE**, ReLU is primarily used in **hidden layers** of deep neural networks to introduce non-linearity and speed up training, while the **output layer** dictates the final activation/loss pairing for the specific task. 

### In summary:

* The **Perceptron** uses a non-differentiable **Step Function** and a simple **Error Update Rule** for linearly separable data.
* **Modern Neural Networks** use a differentiable **Sigmoid** and **Binary Cross-Entropy** for binary classification, and **ReLU** in their hidden layers to learn complex, non-linear patterns.

