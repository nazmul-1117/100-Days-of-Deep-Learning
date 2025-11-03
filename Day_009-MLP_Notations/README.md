# Day_009 | MLP Notations

## General Notation
### Biases
$$
b_{ij}
$$

Where:
* $i$ → Layer's Number
* $j$ → Node's Number

### Output
$$
o_{ij}
$$

Where:
* $i$ → Layer's Number
* $j$ → Node's Number
  

### Weights
$$
w^k_{ij}
$$

Where:
* $k$ → Which `Layer` its going (Current `Layer Number`)
* $i$ → From which `Node` its comming (Previous Layer `Node Number`)
* $j$ → Which `Node` its going (Current Layer `Node Number`)

## General Notation for a Neuron

A single neuron's computation involves a weighted sum of its inputs plus a bias, followed by an activation function:

$$
a = \phi\left( \sum_{j} w_j x_j + b \right) = \phi(z)
$$

Where:
* $x_j$: The $j$-th input to the neuron.
* $w_j$: The weight associated with the $j$-th input.
* $b$: The bias term.
* $z$: The **pre-activation** or **weighted input** ($z = \sum_{j} w_j x_j + b$).
* $\phi(\cdot)$: The **activation function** (e.g., ReLU, Sigmoid, Tanh).
* $a$: The **activation** (output) of the neuron.

## Vectorized Notation for a Layer (Forward Propagation)

To describe the computation for an entire layer efficiently, matrix-vector notation is used. Let $L$ be the index for the layer, with $L=1$ being the input layer, $L=2$ the first hidden layer, and so on.



### 1. Pre-activation (Weighted Input) for Layer $L$:

$$
\mathbf{z}^{[L]} = \mathbf{W}^{[L]} \mathbf{a}^{[L-1]} + \mathbf{b}^{[L]}
$$

Where:
* $\mathbf{z}^{[L]}$: The **pre-activation vector** for layer $L$. Each element is the weighted sum plus bias for a neuron in that layer.
* $\mathbf{W}^{[L]}$: The **weight matrix** for layer $L$. It contains all weights connecting layer $L-1$ to layer $L$.
* $\mathbf{a}^{[L-1]}$: The **activation vector** (output) of the *previous* layer, $L-1$. For the first hidden layer $(L=2)$, $\mathbf{a}^{[1]}$ is the input vector $\mathbf{x}$.
* $\mathbf{b}^{[L]}$: The **bias vector** for layer $L$.

### 2. Activation (Output) for Layer $L$:

$$
\mathbf{a}^{[L]} = \phi^{[L]}\left( \mathbf{z}^{[L]} \right)
$$

Where:
* $\mathbf{a}^{[L]}$: The **activation vector** (output) of layer $L$. This becomes the input for the next layer.
* $\phi^{[L]}(\cdot)$: The layer-specific **activation function**, applied element-wise to the vector $\mathbf{z}^{[L]}$.

***

## Specific Components and Indices

| Term | Notation (Scalar) | Notation (Vector/Matrix) | Description |
| :--- | :--- | :--- | :--- |
| **Input** | $x_i$ | $\mathbf{x}$ or $\mathbf{a}^{[1]}$ | Input features. $\mathbf{a}^{[1]}$ is the activation of the input layer. |
| **Layer Index** | $l$ | $|$ | $l \in \{1, 2, \dots, L\}$. $|=1$ is input, $| > 1$ are hidden/output layers. |
| **Neuron Index**| $i$ | $|$ | Index for a specific neuron in a layer. |
| **Weight** | $W_{i, j}^{[l]}$ | $\mathbf{W}^{[l]}$ | Weight connecting neuron $j$ in layer $l-1$ to neuron $i$ in layer $l$. |
| **Bias** | $b_i^{[l]}$ | $\mathbf{b}^{[l]}$ | Bias for neuron $i$ in layer $l$. |
| **Pre-activation**| $z_i^{[l]}$ | $\mathbf{z}^{[l]}$ | Weighted sum of inputs plus bias for neuron $i$ in layer $l$. |
| **Activation** | $a_i^{[l]}$ | $\mathbf{a}^{[l]}$ | Output of neuron $i$ in layer $l$. |
| **Activation Function** | $\phi(\cdot)$ or $f(\cdot)$ | $\phi^{[l]}(\cdot)$ | Non-linear function applied element-wise in layer $l$. |

## Example for a 3-Layer MLP (Input, 1 Hidden, Output)

Let $x = \mathbf{a}^{[1]}$ be the input.

**Hidden Layer (Layer $L=2$):**

1.  **Pre-activation:** $\mathbf{z}^{[2]} = \mathbf{W}^{[2]} \mathbf{x} + \mathbf{b}^{[2]}$
2.  **Activation:** $\mathbf{a}^{[2]} = \phi^{[2]}(\mathbf{z}^{[2]})$

**Output Layer (Layer $L=3$):**

1.  **Pre-activation:** $\mathbf{z}^{[3]} = \mathbf{W}^{[3]} \mathbf{a}^{[2]} + \mathbf{b}^{[3]}$
2.  **Final Output:** $\mathbf{y} = \mathbf{a}^{[3]} = \phi^{[3]}(\mathbf{z}^{[3]})$


## ✅ Corrected and Polished Version

**General formula for neuron output:**
$$
[
a_j^{(l)} = g \left( \sum_i W_{ji}^{(l)} a_i^{(l-1)} + b_j^{(l)} \right)
]
$$

Where:

* $(a_j^{(l)})$: output (activation) of neuron (j) in layer (l)
* $(g(\cdot))$: activation function (e.g., sigmoid, ReLU, tanh)
* $(W_{ji}^{(l)})$: weight from neuron (i) in layer (l-1) **to** neuron (j) in layer (l)
* $(b_j^{(l)})$: bias for neuron (j) in layer (l)

---

## 🧭 Notation Summary

|      **Component**      |        **Common Notation**       |                                                                 **Meaning of Indices**                                                                 |
| :---------------------: | :------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------: |
| **Output / Activation** |    $(a_j^{(l)}) or (O_j^{(l)})$    |                         Superscript ((l)): layer number (1 = first hidden layer). Subscript (j): neuron index within layer (l).                        |
|        **Weight**       | $(W_{ji}^{(l)}) or (w_{ji}^{(l)})$ | Superscript ((l)): layer **to** which the weights connect. Subscripts (j,i): (j) = target neuron (in layer (l)), (i) = source neuron (in layer (l-1)). |
|         **Bias**        |            $(b_j^{(l)})$          |                                       Superscript ((l)): layer number. Subscript (j): neuron index in that layer.                                      |

---

## 💡 Vectorized (Matrix) Form

For layer (l):
$$
[
z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}
]
\\
[
a^{(l)} = g(z^{(l)})
]
$$

Where:

* $(z^{(l)})$: vector of weighted sums (pre-activations) for layer (l)
* $(W^{(l)})$: weight matrix connecting layer (l-1) to (l)
* $(a^{(l-1)})$: activation vector from the previous layer
* $(b^{(l)})$: bias vector for layer (l)
* $(g(\cdot))$: activation function applied elementwise

---

✅ **Note on Dimensions**

If layer $(l-1)$ has $(n_{l-1})$ neurons and layer $(l)$ has $(n_l)$ neurons:
$$
* (W^{(l)} \in \mathbb{R}^{n_l \times n_{l-1}})
* (b^{(l)} \in \mathbb{R}^{n_l})
* (a^{(l)} \in \mathbb{R}^{n_l})
$$