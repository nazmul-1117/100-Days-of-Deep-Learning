# Day_032 | Optimizers in Deep Learning | Part-1 | Why Need Advance Optimizer
Optimizers are algorithms or methods used to change the attributes of your neural network, such as **weights** and **biases**, to minimize the **Loss Function**. They are crucial for training efficient and accurate deep learning models.

---

## 📉 Problem with Traditional Gradient Descent (SGD)

While **Stochastic Gradient Descent (SGD)** is the foundational optimization algorithm, it suffers from several critical drawbacks that make training deep, complex networks slow and difficult:

1.  **Slow and Oscillatory Convergence:** When the loss surface is **non-spherical** (i.e., highly elongated or has plateaus), SGD struggles. It has to take small steps in the shallow direction to avoid overshooting the minimum in the steep direction, resulting in a **slow "zig-zag" path** to convergence.
2.  **Sensitivity to Learning Rate:** Choosing the correct **learning rate ($\eta$)** is difficult. A rate that's too small leads to painfully slow convergence, while one that's too large causes oscillations around the minimum or even divergence.
3.  **Stuck in Local Minima/Saddle Points:** SGD uses the same learning rate for all parameters throughout the training. If it encounters a **saddle point** (a flat region with multiple dimensions) or a shallow local minimum, it can get stuck or take an excessively long time to navigate them.

---

Traditional Gradient Descent methods — **Batch Gradient Descent, Mini-Batch Gradient Descent, and Stochastic Gradient Descent (SGD)** — are foundational, but each suffers from important limitations that motivated the development of more advanced optimizers (like Momentum, RMSProp, Adam, etc.).
Here is a clear breakdown of the problems with each:

---

## ✅ **1. Batch Gradient Descent (Full-batch GD)**

**Definition:** Uses the *entire* training dataset to compute the gradient at every update.

### **Problems**

### ❌ *Very slow for large datasets*

You must process all samples before a single update → extremely inefficient for modern datasets with millions of samples.

### ❌ *High computational + memory cost*

Storing and operating on a full dataset each step may be impossible for big data.

### ❌ *Can get stuck in local minima or saddle points*

It updates in a single direction deterministically and may not escape problematic regions.

### ❌ *Poor generalization in noisy landscapes*

Because it uses the mean gradient, it may fail when the loss surface has many fluctuations (common in deep nets).

---

## ✅ **2. Stochastic Gradient Descent (SGD)**

**Definition:** Uses **one random sample** to compute the gradient at each step.

### **Problems**

### ❌ *Very high gradient noise (too much variance)*

One sample gives an extremely noisy gradient estimate → causes:

* unstable training
* fluctuating loss curves
* difficulty converging to minima
* smaller effective learning rates

### ❌ *May “bounce around” minima instead of settling*

Because updates are noisy, SGD often oscillates and may not converge without learning-rate decay.

### ❌ *Sensitive to learning rate*

A slightly too-large learning rate can cause divergence.

---

## ✅ **3. Mini-Batch Gradient Descent**

**Definition:** Uses a small batch (e.g., 32, 64, 128) to compute gradients.
This is what most training uses today.

### **Problems**

### ❌ *Still noisy, though less than SGD*

Variance is reduced but still present → convergence may be slow or unstable.

### ❌ *Learning-rate sensitivity remains*

Choosing the right learning rate is still difficult. Too high → divergence; too low → slow training.

### ❌ *May get stuck at saddle points or plateaus*

Mini-batch gradients can still be misaligned in regions where the loss surface is flat.

### ❌ *Batch-size choice affects convergence*

* Too small → noisy
* Too large → slow and generalizes poorly

---

## 🎯 **Why Advanced Optimizers Were Created**

To address these issues, optimizers introduce features like:

✓ **Momentum** → reduces oscillation by smoothing gradients

✓ **RMSProp / AdaGrad** → adaptive learning rates

✓ **Adam** → combines momentum + adaptive learning rates (most widely used)

---

## 🔥 Summary Table

| Optimizer         | Main Problem                                            |
| ----------------- | ------------------------------------------------------- |
| **Batch GD**      | Too slow, high memory cost, poor scalability            |
| **SGD**           | Very noisy gradients, unstable convergence              |
| **Mini-Batch GD** | Still noisy, sensitive learning rate, batch-size issues |