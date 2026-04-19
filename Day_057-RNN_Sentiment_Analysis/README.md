# Day_057 | RNN Sentiment Analysis | RNN Code Example in Keras

-----

## 🧐 RNN Sentiment Analysis

Sentiment Analysis, in this context, is a **binary classification** task: given a sequence of text (like a movie review), the model must classify it as either **Positive** (1) or **Negative** (0).

### The RNN Process

1.  **Tokenization & Embedding:** The raw text is broken down into tokens (words). These tokens are then converted into dense, numerical **vector embeddings**. This converts each word into a point in a high-dimensional space where similar words are closer together.
2.  **Recurrent Layer (Memory):** The sequence of word embeddings is fed into an RNN layer (e.g., LSTM or GRU). The recurrent nature allows the network to build a **contextual representation** of the sentence by maintaining a hidden state (memory) that captures the influence of past words on the current word. For instance, it can understand that the word "not" reverses the meaning of a subsequent positive word like "great."
3.  **Classification:** The final hidden state of the RNN (which summarizes the entire sequence's context) is passed to a final **Dense layer** with a **Sigmoid activation** to output a probability score between 0 and 1 (where 0 is negative and 1 is positive).

-----

## 💻 RNN Code Example in Keras

The best practice today is to use **LSTM (Long Short-Term Memory)** or **GRU (Gated Recurrent Unit)** layers instead of the basic RNN layer, as they solve the **vanishing gradient problem** and are better at capturing long-term dependencies.

This example uses the Keras `Sequential` API and demonstrates the necessary layers for a sequence classification task.

>Keras

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- 1. Hyperparameters and Setup ---
# Assume we have preprocessed and tokenized text data (X_train, y_train)
# Vocabulary size (max number of unique words)
VOCAB_SIZE = 10000 
# Maximum length of input sequences
MAX_LENGTH = 200 
# Dimension of the word embeddings
EMBEDDING_DIM = 128 

# Assume X_train is padded:
# X_train = pad_sequences(tokenizer.texts_to_sequences(raw_text), maxlen=MAX_LENGTH)

# --- 2. Model Definition (Sequential API) ---
model = Sequential([
    # Layer 1: Embedding
    # Turns positive integers (word indices) into dense vectors of fixed size.
    Embedding(input_dim=VOCAB_SIZE, 
              output_dim=EMBEDDING_DIM, 
              input_length=MAX_LENGTH),
    
    # Layer 2: LSTM
    # The recurrent layer. Returns only the final output state for classification.
    LSTM(units=64),
    
    # Layer 3: Dense Hidden Layer (Optional)
    Dense(units=32, activation='relu'),
    
    # Layer 4: Output Layer
    # Single neuron with sigmoid for binary classification (sentiment 0 or 1).
    Dense(units=1, activation='sigmoid')
])

# --- 3. Model Compilation ---
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# --- 4. Model Summary ---
# Prints the structure of the model, showing the massive parameter count 
# in the Embedding and LSTM layers.
model.summary() 

# --- 5. Training (Conceptual) ---
# model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)
```

### Key Layer Roles

  * **`Embedding`:** Responsible for converting the input sequence of word IDs into meaningful dense vectors. The weights of this layer are learned during training.
  * **`LSTM`:** The main engine. It processes the sequence step-by-step. Since we are doing sequence **classification** (outputting one sentiment for the whole review), we typically set `return_sequences=False` (the default), meaning it only returns the final hidden state, which summarizes the entire sequence.
  * **`Dense(1, activation='sigmoid')`:** The classification layer. It converts the final hidden state into the final probability of the review being positive.

---

## ✅ **RNN Sentiment Analysis — Overview**

**Sentiment analysis** classifies text (e.g., reviews, tweets) as *positive*, *negative*, or *neutral*.
An **RNN (Recurrent Neural Network)** processes input *sequentially*, making it suitable for text because words depend on previous context.

However, standard RNNs suffer from vanishing gradients, so in practice **LSTM** or **GRU** layers are used instead — but they are still part of the RNN family.

---

## ✅ **Simple RNN Sentiment Analysis Pipeline**

1. **Load text dataset**
   (IMDB movie review dataset is built into Keras)
2. **Tokenize and pad sequences**
3. **Build an RNN model (SimpleRNN / LSTM / GRU)**
4. **Train the model**
5. **Evaluate and make predictions**

---

## 🧪 **Complete Keras Example: RNN Sentiment Analysis (SimpleRNN + IMDB Dataset)**

> Keras
```python
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# -------------------------------
# 1. Load IMDB dataset
# -------------------------------
num_words = 10000  # keep top 10k words
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

# Pad sequences to same length
max_len = 200
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

# -------------------------------
# 2. Build RNN Model
# -------------------------------
model = Sequential([
    Embedding(input_dim=num_words, output_dim=128, input_length=max_len),
    SimpleRNN(64, return_sequences=False),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------------------------------
# 3. Train Model
# -------------------------------
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

# -------------------------------
# 4. Evaluate
# -------------------------------
loss, acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", acc)
```

---

## 🔁 **LSTM Version (Recommended)**

Replace the SimpleRNN layer with an LSTM:

> Keras
```python
from tensorflow.keras.layers import LSTM

model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=200),
    LSTM(64),
    Dense(1, activation='sigmoid')
])
```

---

## 🧠 **GRU Version (Fast and Effective)**

> Keras
```python
from tensorflow.keras.layers import GRU

model = Sequential([
    Embedding(10000, 128),
    GRU(64),
    Dense(1, activation='sigmoid')
])
```

---