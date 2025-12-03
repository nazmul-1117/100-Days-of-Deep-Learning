# Day_030 | Xavier/Glorat And He Weight Initialization in Deep Learning

## ⭐ 1. Why Weight Initialization Matters

When weights are not initialized properly:

* Activations explode or vanish as they pass through layers
* Gradients explode or vanish backward
* Training becomes unstable or completely fails
* Deep networks cannot learn meaningful features

Modern initialization techniques like **Xavier (Glorot)** and **He (Kaiming)** solve these issues.

---

## ⭐ 2. Xavier / Glorot Initialization

**Introduced by Glorot & Bengio (2010).**
Designed for **sigmoid**, **tanh**, and **linear** activations.

---

### ✅ **Goal**

Keep the **variance** of activations and gradients **constant across layers**.

Without this, deeper layers get:

* Vanishing activations (too small)
* Exploding activations (too large)

---

### 🔍 **Idea Behind Xavier**

For a layer with:

* ( fan_{in} ) = number of input neurons
* ( fan_{out} ) = number of output neurons

We want:

$$
Var(W) = \frac{2}{fan_{in} + fan_{out}}
$$

---

## ✅ **Xavier Uniform**


$$
\text{[-limit, +limit]}\\
$$

$$
\text{limit} = \sqrt{\frac{6}{fan_{in}+fan_{out}}}
$$

$$
W \sim U\left(-\sqrt{\frac{6}{fan_{in}+fan_{out}}}, +\sqrt{\frac{6}{fan_{in}+fan_{out}}}\right)
$$

## ✅ **Xavier Normal**

$$
\sqrt{\frac{1}{fan_{in}}}
$$

Another,

$$
W \sim N\left(0, \frac{2}{fan_{in}+fan_{out}}\right)
$$

---

## ⭐ When to Use Xavier Initialization

Use for **activation functions with values in both negative and positive ranges**:

| Activation  | Xavier Recommended? |
| ----------- | ------------------- |
| Sigmoid     | ✔️ Yes              |
| Tanh        | ✔️ Yes              |
| Softsign    | ✔️ Yes              |
| Linear      | ✔️ Yes              |
| ReLU family | ❌ No (use He)       |

---

## ⭐ Advantages

* Prevents vanishing/exploding activations in non-ReLU networks
* Stable convergence
* Works well for fully connected and CNN layers using tanh/sigmoid

## ❌ Disadvantages

* Not ideal for ReLU, which discards half of activations → needs different scaling
* Still may lead to saturation in deep sigmoid networks

---

## ⭐ 3. He Initialization (Kaiming Initialization)

**Introduced by He et al. (2015).**
Designed **specifically for ReLU & its variants**.

---

## 🔍 Why Xavier Fails for ReLU

ReLU zeroes out negative activations:

$$
f(x) = \max(0, x)
$$

That means **only half** of the inputs propagate forward ⇒ variance is reduced by **1/2**.

---

## 🔥 He Initialization Fixes This

To compensate for halved variance, He increases initialization variance:

$$
Var(W) = \frac{2}{fan_{in}}
$$

---

## ✅ He Normal

$$
W \sim N\left(0, \frac{2}{fan_{in}}\right)
$$

## ✅ He Uniform

$$
\text{[-limit, +limit]}\\
$$

$$
\text{limit} = \sqrt{\frac{6}{fan_{in}}}
$$

$$
W \sim U\left(-\sqrt{\frac{6}{fan_{in}}}, +\sqrt{\frac{6}{fan_{in}}}\right)
$$

---

## ⭐ When to Use He Initialization

Ideal for **ReLU and its smooth variants**:

| Activation | He Recommended?                 |
| ---------- | ------------------------------- |
| ReLU       | ✔️ Yes                          |
| Leaky ReLU | ✔️ Yes                          |
| PReLU      | ✔️ Yes                          |
| GELU       | ✔️ Yes                          |
| ELU        | ✔️ Often good                   |
| Swish      | ✔️ Sometimes better than Xavier |

---

## ⭐ Advantages

* Best performance for deep CNNs
* Prevents dying ReLU (better gradient flow)
* Helps very deep architectures converge faster

## ❌ Disadvantages

* Not suitable for tanh or sigmoid
* If learning rate is large, can cause exploding gradients in ELU/SELU

---

## ⭐ 4. Comparison: Xavier vs He Initialization

| Feature           | Xavier / Glorot                  | He / Kaiming                        |
| ----------------- | -------------------------------- | ----------------------------------- |
| Designed for      | Sigmoid, Tanh                    | ReLU family                         |
| Variance          | $( \frac{2}{fan_{in}+fan_{out}} )$ | $( \frac{2}{fan_{in}} )$              |
| Problem Addressed | Vanishing/exploding activations  | Lost variance in ReLU negative side |
| Deep CNNs         | ❌ Weak                           | ✔️ Strong                           |
| Transformers      | Sometimes                        | ✔️ Often (GELU/Swish)               |
| RNNs (tanh)       | ✔️                               | ❌                                   |

---

## ⭐ 5. Practical Guidelines (Easy to Remember)

### ✔ Use **He Initialization** if your activation is:

ReLU, LeakyReLU, PReLU, GELU, Swish, ELU

### ✔ Use **Xavier Initialization** if your activation is:

Tanh, Sigmoid, Softmax, Linear

### ✔ Use **LeCun Initialization** if activation is:

SELU (“self-normalizing neural networks”)

---

## ⭐ 6. PyTorch Examples

### Xavier:

```python
torch.nn.init.xavier_uniform_(layer.weight)
torch.nn.init.xavier_normal_(layer.weight)
```

### He:

```python
torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
```

---

## ⭐ 7. TensorFlow / Keras Examples

### Xavier:

```python
kernel_initializer=tf.keras.initializers.GlorotUniform()
```

### He:

```python
kernel_initializer=tf.keras.initializers.HeNormal()
```

---

## ⭐ 8. Summary in One Line

* **Xavier** keeps variance stable for **sigmoid/tanh**
* **He** keeps variance stable for **ReLU** by doubling the scale to handle the half-zero outputs
