# Day 007 | Major Problems with the Single Perceptron

### 1. The XOR Problem (Non-Linear Separability)
This is the most famous limitation, first highlighted by Marvin Minsky and Seymour Papert in their 1969 book *Perceptrons*.

* **The Issue:** A single-layer perceptron is a **linear classifier**. It can only learn a **decision boundary** that is a straight line (or a flat hyperplane in higher dimensions).
* **The Consequence:** It can successfully model linearly separable functions like the **AND** gate and the **OR** gate, where a single straight line can separate the True/False outputs. However, it cannot solve the **Exclusive OR (XOR)** gate, where the data points for the two classes cannot be separated by a single straight line.
    * 
* **Solution:** This led to the development of **Multi-Layer Perceptrons (MLPs)**, which introduce **hidden layers** with non-linear activation functions. A network with just one hidden layer can solve the XOR problem.

### 2. Output Limitations (Non-Differentiable)
The perceptron's activation and training mechanisms are limited compared to modern neural networks.

* **Hard Threshold:** It uses the **Step Function** as its activation function, which only outputs a binary value (0 or 1, or $-1$ or $1$). This hard output prevents the model from generating more nuanced outputs like **probabilities**.
* **Non-Differentiable Loss:** The Step Function is not **differentiable** at the threshold. In calculus terms, you can't calculate its slope (derivative) across the entire range. This means you cannot use the efficient, modern optimization technique called **Gradient Descent** (and the **Backpropagation** algorithm) to train the model, which is essential for deep learning. Instead, it relies on its simple, error-driven **Perceptron Trick** update rule.

### 3. Convergence Guarantee
While the Perceptron Convergence Theorem is a powerful result, it comes with a major caveat:

* **Conditional Convergence:** The algorithm is **only guaranteed to converge** and find a perfect solution if the training data is **already linearly separable**. If the data is not separable, the perceptron learning algorithm will enter an infinite loop, constantly trying to adjust the weights without ever finding a perfect boundary.
## Images
![images](assets/bitwise_datasets.png)