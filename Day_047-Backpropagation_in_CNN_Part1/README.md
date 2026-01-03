# Day_047 | 📐 Backpropagation in CNN | Part 1 | Fully Connected Layer

We will focus on the gradient calculation for the weights and biases of the final **Fully Connected (FC)** layers in a CNN's classification block.

---

## 📐 Backpropagation in the Fully Connected Layers of a CNN

In a CNN, the output of the final pooling layer is **flattened** into a 1D vector and then fed into a sequence of standard ANN (Dense) layers for final classification. Backpropagation for these layers proceeds exactly as it would in an MLP.

Let's assume the final layers are:
* $L$: The **Output Layer** (e.g., Softmax for classification).
* $L-1$: The last **Hidden Fully Connected Layer**.

### 1. Notation

| Term | Notation | Description |
| :--- | :--- | :--- |
| **Layer Index** | $l$ | Current layer (e.g., $l=L$ or $l=L-1$). |
| **Activation** | $\mathbf{a}^{(l)}$ | Output vector of layer $l$. $\mathbf{a}^{(L-1)}$ is the input to the output layer. |
| **Weighted Input** | $\mathbf{z}^{(l)}$ | Pre-activation input to layer $l$: $\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$. |
| **Loss Function** | $J$ | The final loss value (e.g., Cross-Entropy). |
| **Error Signal** | $\boldsymbol{\delta}^{(l)}$ | The gradient of the loss with respect to the weighted input $\mathbf{z}^{(l)}$. |
| **Weights** | $\mathbf{W}^{(l)}$ | Weight matrix connecting layer $l-1$ to layer $l$. |
| **Biases** | $\mathbf{b}^{(l)}$ | Bias vector for layer $l$. |

---

### 2. Step 1: Calculate the Output Layer Error Signal ($\boldsymbol{\delta}^{(L)}$)

This is the starting point. It requires finding the derivative of the loss function with respect to the pre-activation input of the output layer.

$$
\boldsymbol{\delta}^{(L)} = \frac{\partial J}{\partial \mathbf{z}^{(L)}} = \frac{\partial J}{\partial \mathbf{a}^{(L)}} \odot g'(\mathbf{z}^{(L)})
$$

* $g'(\mathbf{z}^{(L)})$ is the derivative of the output activation function (e.g., Softmax or Sigmoid).
* The $\odot$ denotes the Hadamard product (element-wise multiplication).

> **Simplified for Common Loss/Activation Pairs:**
> * **Cross-Entropy Loss with Softmax Output:** The error signal simplifies to:
>     $$\boldsymbol{\delta}^{(L)} = \mathbf{a}^{(L)} - \mathbf{y}$$
>     Where $\mathbf{y}$ is the one-hot encoded true label and $\mathbf{a}^{(L)}$ is the predicted probability vector.

---

### 3. Step 2: Backpropagate the Error Signal ($\boldsymbol{\delta}^{(L-1)}$)

The error signal from the output layer ($\boldsymbol{\delta}^{(L)}$) must be propagated backward to calculate the error signal for the previous hidden layer ($L-1$).

$$
\boldsymbol{\delta}^{(L-1)} = \left( (\mathbf{W}^{(L)})^T \boldsymbol{\delta}^{(L)} \right) \odot g'(\mathbf{z}^{(L-1)})
$$

* $(\mathbf{W}^{(L)})^T \boldsymbol{\delta}^{(L)}$: The error is weighted by the transpose of the connection weights $\mathbf{W}^{(L)}$. This is the crucial step that distributes the output error back to the neurons in the previous layer.
* $g'(\mathbf{z}^{(L-1)})$ (e.g., derivative of ReLU or Tanh): Applies the derivative of the hidden layer's activation function.

This step is repeated for all preceding fully connected layers until the flattened layer is reached.

### ANN Part
> Some Equations

$$L = -[y \log(\hat{y}) + (1 - y)\log(1 - \hat{y})]$$

$$
Z_2 = W_2F + b_2
$$

$$
A_2 = \sigma(Z_2)
$$

Derivative of loss w.r.t. output:
> Calculate W

$$
\frac{\partial L}{\partial w_2} = \frac{\partial L}{\partial A_2} \text{x} \frac{\partial A_2}{\partial Z_2} \text{x} \frac{\partial Z_2}{\partial W_2}
$$

> Calculate b

$$
\frac{\partial L}{\partial w_2} = \frac{\partial L}{\partial A_2} \text{x} \frac{\partial A_2}{\partial Z_2} \text{x} \frac{\partial Z_2}{\partial b}
$$


### Calculate $\frac{\partial L}{\partial A_2}$
> A_2 -> a2

$$
\frac{\partial L}{\partial A_2} = \frac{\partial L}{\partial a_2}
$$

$$
\frac{\partial L}{\partial A_2} = \frac{\partial}{\partial a_2}
 [-[y \log(\hat{y}) + (1 - y)\log(1 - \hat{y})]]
$$

$$
\frac{\partial L}{\partial A_2} = -\frac{y}{a_2} +\frac{1-y}{1-a_2}
$$

$$
\frac{\partial L}{\partial A_2} = \frac{a_2-y}{a_2(1-a_2)}
$$


### Calculate $\frac{\partial A_2}{\partial Z_2}$

$$
\frac{\partial A_2}{\partial Z_2} = \sigma (z_2) [1 - \sigma (z_2)]
$$

$$
\frac{\partial A_2}{\partial Z_2} = a_2 \cdot (1 - a_2)
$$


### Calculate $\frac{\partial Z_2}{\partial w_2}$

$$
\frac{\partial Z_2}{\partial w_2} = F
$$


### Calculate $\frac{\partial Z_2}{\partial b}$

$$
\frac{\partial Z_2}{\partial b} = 1
$$


### Final Calculate $\frac{\partial L}{\partial w_2}$

$$
\frac{\partial L}{\partial w_2} = F \cdot(A_2 - y)
$$


### Final Calculate $\frac{\partial L}{\partial b}$

$$
\frac{\partial L}{\partial b} = (A_2 - y)
$$

---

### 4. Step 3: Calculate Gradients for Weights and Biases

Once the error signal $\boldsymbol{\delta}^{(l)}$ is known for a layer $l$, the gradients for the weights and biases of that layer are calculated using the chain rule:

#### A. Gradient for Weights ($\frac{\partial J}{\partial \mathbf{W}^{(l)}}$)

The gradient of the loss with respect to the weights is the outer product of the error signal $\boldsymbol{\delta}^{(l)}$ and the activation from the previous layer $\mathbf{a}^{(l-1)}$:

$$
\frac{\partial J}{\partial \mathbf{W}^{(l)}} = \boldsymbol{\delta}^{(l)} (\mathbf{a}^{(l-1)})^T
$$

#### B. Gradient for Biases ($\frac{\partial J}{\partial \mathbf{b}^{(l)}}$)

The gradient of the loss with respect to the biases is simply the error signal itself:

$$
\frac{\partial J}{\partial \mathbf{b}^{(l)}} = \boldsymbol{\delta}^{(l)}
$$

---

### 5. Step 4: Parameter Update (Gradient Descent)

Finally, the calculated gradients are used to update the weights and biases using the chosen optimizer (e.g., Adam, SGD with Momentum) and learning rate ($\eta$):

$$
\mathbf{W}^{(l)}_{\text{new}} = \mathbf{W}^{(l)}_{\text{old}} - \eta \frac{\partial J}{\partial \mathbf{W}^{(l)}}
$$

$$
\mathbf{b}^{(l)}_{\text{new}} = \mathbf{b}^{(l)}_{\text{old}} - \eta \frac{\partial J}{\partial \mathbf{b}^{(l)}}
$$