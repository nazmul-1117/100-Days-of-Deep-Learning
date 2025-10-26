# Day 001 | Course Raodmap | 📝 Deep Learning Study Notes

## Phase 1: Fundamentals (ANNs & Core Mechanics)

### 1\. The Building Blocks

| Concept | Definition/Analogy | Key Code Components |
| :--- | :--- | :--- |
| **Perceptron/Neuron** | The core unit: takes inputs, multiplies by weights, adds bias, applies activation. | `tf.keras.layers.Dense(...)` |
| **Activation Functions** | Introduce **non-linearity**. Allows the network to learn complex patterns (e.g., XOR). **ReLU** is the standard. | `activation='relu'`, `activation='sigmoid'` |
| **Loss Function** | Measures "how wrong" the model is. We aim to minimize this. (e.g., **Cross-Entropy** for classification). | `loss='categorical_crossentropy'` |
| **Gradient Descent** | The "optimizer." Steers the weights downhill (opposite the gradient) to find the minimum loss. | `optimizer='sgd'` or `optimizer='adam'` |
| **Backpropagation** | The efficient algorithm for calculating the **gradient** by applying the **Chain Rule** backward through the network. | *Implicitly handled* by frameworks like Keras/PyTorch. |

<br>

### 2\. Training and Regularization

| Concept | Goal/Function | Practical Application |
| :--- | :--- | :--- |
| **Learning Rate ($\alpha$)** | Size of the step taken during gradient descent. Too large $\rightarrow$ instability; too small $\rightarrow$ slow training. | Use **Adam** as a self-tuning optimizer. |
| **Batch Normalization** | Normalizes the output of a hidden layer. Stabilizes and dramatically speeds up training. | `tf.keras.layers.BatchNormalization()` |
| **Dropout** | Randomly sets a fraction of neurons to zero during training. Prevents neurons from co-adapting, acting as a powerful **regularizer** against overfitting. | `tf.keras.layers.Dropout(0.2)` (usually 20%) |
| **Overfitting** | Model learns the training data noise; fails on unseen data. | Use **Dropout** and **L2 Regularization**. |

-----

## Phase 2: CNNs (Computer Vision)

### 3\. Convolutional Networks

| Layer/Concept | Purpose | How it works (The intuition) |
| :--- | :--- | :--- |
| **Convolution** | Learns local patterns (edges, shapes, textures) by applying a **filter** (kernel) across the input image. | Like a magnifying glass scanning for specific features. |
| **Pooling (Max/Avg)** | Downsamples the input volume. Reduces parameters and makes the learned features slightly location-invariant. | "Compressing" the image to keep the most important information. |
| **Transfer Learning** | Using weights from a model trained on a massive dataset (like **ImageNet**) as a starting point for your task. | "Standing on the shoulders of giants." Crucial for smaller datasets. |
| **Architectures** | **ResNet**: Uses **Skip Connections** (Residual Blocks) to train very deep networks effectively. | Solves the *degradation problem* in deep networks. |

-----

## Phase 3: RNNs & Transformers (Sequence Data/NLP)

### 4\. Sequential Data Models

| Model Type | Best For | Key Mechanism |
| :--- | :--- | :--- |
| **RNN** | Basic sequence data, short texts. | Has a **hidden state** (memory) passed from one time step to the next. |
| **LSTM / GRU** | Long sequences (long-term dependencies). Solves the **Vanishing Gradient** problem in basic RNNs. | Uses internal **Gates** (Forget, Input, Output) to regulate information flow. |
| **Word Embeddings** | Converts text (discrete words) into dense, continuous vectors that capture semantic meaning. | Allows the model to understand the *relationship* between words (e.g., "king" is near "queen"). |

<br>

### 5\. The Attention Revolution

| Model Type | Core Innovation | Impact |
| :--- | :--- | :--- |
| **Attention** | Allows the model to dynamically focus on the most relevant parts of the input sequence when making a prediction. | Better handling of long sequences; much more contextually aware. |
| **Transformer** | Entirely replaces recurrence (RNNs) with **Self-Attention**. Faster, more parallelizable training. | The foundation of all modern Large Language Models (**LLMs**) like GPT and BERT. |
| **BERT / GPT** | Large, pre-trained Transformer models. | **BERT** is great for *understanding* text (encoding); **GPT** is great for *generating* text (decoding). |

-----

## Phase 4: Generative & Specialized Models

### 6\. Creative AI

| Model Type | Goal/Task | Key Components |
| :--- | :--- | :--- |
| **Autoencoders** | Learn a compressed representation (latent space) of the input data. Useful for dimensionality reduction or **Anomaly Detection**. | **Encoder** (compression) and **Decoder** (reconstruction). |
| **GANs** | Generate new, realistic data samples (images, audio, text). | A competition between a **Generator** (the faker) and a **Discriminator** (the cop). |

-----

## 7\. Next Steps (Focus Areas)

1.  **Framework Choice:** Become proficient in **PyTorch** for flexibility or **TensorFlow** for production.
2.  **MLOps:** Learn how to deploy models using **Docker** and serve them via **FastAPI** or cloud services.
3.  **Advanced DRL:** Explore **Deep Q-Networks (DQN)** for decision-making systems (e.g., game AI).