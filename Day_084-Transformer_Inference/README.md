# Day_084 | ⚡ Transformer Inference (Sequence Generation)

Transformer inference, particularly for generative tasks like machine translation or text completion, is the process of generating the output sequence one token at a time using the trained model. This process is inherently **sequential** (autoregressive), unlike the parallel nature of training.

The inference process focuses entirely on the **Decoder** and involves a critical loop where the model's output is fed back as its input for the next step.

---

## 1. The Inference Procedure

The goal is to generate the output sequence $\mathbf{Y} = (y_1, y_2, \dots, y_T)$ given the input $\mathbf{X}$.

### Step 1: Pre-Process the Encoder Input (One-Time Setup)

1.  **Encoder Pass:** The entire input sequence $\mathbf{X}$ is fed through the **Encoder stack**.
2.  **Context Extraction:** The Encoder processes the input entirely in parallel and outputs a sequence of final, contextualized hidden states ($\mathbf{H}_{\text{enc}}$).
3.  **Storage:** This output ($\mathbf{H}_{\text{enc}}$) is stored and passed to **all Cross-Attention sub-layers** of the Decoder for every subsequent decoding step.

### Step 2: The Autoregressive Decoding Loop

The Decoder starts the loop, generating one token per iteration until the sequence is complete.

| Iteration | Decoder Input | Action |
| :--- | :--- | :--- |
| **$t=1$** | **Start Token** (`<START>`) | **Decoder Pass:** The Decoder runs its three sub-layers (Masked Self-Attention, Cross-Attention, FFN). The Masked Self-Attention only sees the `<START>` token (since the target sequence is only one token long). **Prediction:** The Softmax layer outputs a probability distribution; the token with the highest probability ($y_1$) is selected. |
| **$t=2$** | **$<$START$>$ + $y_1$** | **Decoder Pass:** The Decoder runs again. The Masked Self-Attention now processes two tokens ($\mathbf{y}_{1}$ and $\mathbf{y}_{2}$). The mask ensures that the computation for $y_2$ only attends to $y_1$. **Prediction:** The model generates the next token $y_2$. |
| **$t=T$** | **$<$START$>$ + $y_1 \dots y_{T-1}$** | **Loop Continuation:** This process repeats, with the entire sequence generated so far being fed back into the decoder at each step. **End Condition:** The loop stops when the model predicts the **End-of-Sequence token** (`<END>`). |

## 2. Speed and Parallelism

### Training vs. Inference

| Aspect | Training (Fast) | Inference (Slow) |
| :--- | :--- | :--- |
| **Sequence Processing** | **Fully Parallel:** Target sequence processed all at once via **Masked Self-Attention**. | **Sequential/Autoregressive:** Token $t$ must be generated before token $t+1$. |
| **Cause of Slowness** | The decoder cannot reuse computations from the previous step; it must **re-run the full sequence** (from `<START>` to $y_{t-1}$) in the Self-Attention block at every step. |

### Optimization (Key-Value Caching)

The main inefficiency during inference is the repeated calculation of $\mathbf{K}$ and $\mathbf{V}$ matrices for the already-generated tokens (e.g., in step 5, we recalculate $\mathbf{K}$ and $\mathbf{V}$ for $y_1$ through $y_4$, which were already calculated in previous steps).

To speed this up, modern implementations use **Key-Value Caching**:

* The Keys ($\mathbf{K}$) and Values ($\mathbf{V}$) from the previous decoding steps are **saved** (cached).
* In the next step, only the $\mathbf{K}$ and $\mathbf{V}$ for the **new token** are calculated, and the new ones are simply appended to the cache.
* This significantly reduces the repeated computational complexity of the **Self-Attention** block during decoding, making inference much faster.

## 3. Prediction Techniques (Decoding Strategies)

The final prediction step often uses more sophisticated techniques than simply picking the highest probability token (greedy search) to achieve higher quality results:

* **Greedy Search:** Selects the token with the highest probability at each step. It is fast but prone to errors (gets stuck in local optima).
* **Beam Search:** Explores a fixed number ($k$) of the most promising paths (sequences) simultaneously. It keeps track of the $k$ partial sequences with the highest total probability score and generates the next token for all $k$ sequences. This yields higher quality but is computationally slower.


---

## **How Inference is Done in Transformer Models**

### **1. Introduction**

Transformer architectures have become the foundation of modern deep learning systems, powering large language models (LLMs), machine translation, vision transformers, and multimodal models. While training mechanisms like self-attention and backpropagation are widely discussed, **inference**—how the model generates predictions after being trained—is equally critical.
This document provides a detailed, technical, and practical explanation of how inference works inside a Transformer.

---

## **2. Overview of a Transformer at Inference Time**

During inference, the Transformer uses its learned weights to process new input data without updating any parameters.
Key steps include:

1. **Tokenization & Embedding**
2. **Passing tokens through Encoder/Decoder stacks**
3. **Attention computations (Self-Attention & Cross-Attention)**
4. **Generation strategy (Greedy, Beam Search, Sampling)**
5. **Iterative token generation (auto-regression)**
6. **Stopping criteria**

While training processes entire sequences in parallel, **inference in decoder-only or seq2seq models is mostly sequential** because each generated token depends on previously generated tokens.

---

## **3. Tokenization and Input Processing**

### **3.1 Tokenization**

Input text is split using a tokenizer such as:

* BPE (Byte Pair Encoding)
* WordPiece
* SentencePiece

The output is a sequence of token IDs.

**Example:**
“Transformers are powerful” → `[2034, 55, 19483, …]`

### **3.2 Embedding Lookup**

Each token ID is mapped to a learned vector called **token embedding**.

The model adds:

* **Token Embedding**
* **Positional Embedding** (absolute or rotary)

Result:
`X = TokenEmbedding + PositionalEncoding`

This becomes the input to the first Transformer block.

---

## **4. Inference in Encoder-Only, Decoder-Only, and Encoder-Decoder Models**

### **4.1 Encoder-Only Models (e.g., BERT)**

Used for classification, retrieval, NLU tasks.

* Uses **bidirectional self-attention**
* Processes whole input sequence in parallel
* Outputs contextual representations for each token or a special CLS token

🟦 **Inference is fast because there's no token-by-token processing.**

---

### **4.2 Decoder-Only Models (e.g., GPT family, Llama, Mistral)**

Used for **text generation**.

* Uses **causal (masked) self-attention**
* Cannot look at future tokens
* Generates tokens **one by one**
* Output token at step *t* depends only on `[0 … t-1]`

🟥 **Inference is sequential and expensive** for long sequences.

---

### **4.3 Encoder–Decoder Models (e.g., T5, original Transformer)**

Used for machine translation and seq2seq tasks.

Inference consists of:

1. **Encoder processes input sequence (parallel)**
2. **Decoder generates output one token at a time**, attending to:

   * previous outputs (self-attention)
   * encoder outputs (cross-attention)

---

## **5. Attention Mechanics in Inference**

### **5.1 Computing Q, K, V**

In each layer, the model computes:

```
Q = XW_Q
K = XW_K
V = XW_V
```

### **5.2 Self-Attention Score**

```
scores = softmax( (QKᵀ) / sqrt(d_k) )
```

### **5.3 Masking During Decoder Inference**

A **causal mask** hides future positions.
This ensures the model does not peek ahead.

### **5.4 Caching Key/Value for Faster Inference**

**One of the most important differences from training:**

➡️ **During inference, we cache K and V vectors for previously generated tokens.**
This avoids recomputing attention for the entire sequence at every step.

This is called:

* **KV caching**
* **Attention caching**
* **Past key/value states**

Caching results in massive speed improvements:

* Without caching: O(n²) per step
* With caching: O(n) per step

---

## **6. How a Token is Generated Step-by-Step (Decoder-Only Example)**

Suppose we have already produced tokens `[t₁, t₂, … tₙ]`.
To generate the next token `tₙ₊₁`, the model:

1. Takes only the latest token `tₙ`
2. Embeds it → vector Xₙ
3. Runs Xₙ through each layer
4. Uses cached K,V for previous tokens
5. Computes attention between:

   * new Qₙ
   * cached K[1:n] and V[1:n]
6. Passes through feed-forward network
7. Produces logits (vocabulary-sized scores)
8. Applies a generation strategy (softmax → sample → pick)

This repeats until a stop condition:

* EOS token
* max length reached
* end-of-conversation heuristic

**This is called *auto-regressive generation*.**

---

## **7. Generation Strategies**

### **7.1 Greedy Search**

Pick the token with maximum probability.

Pros: Simple, deterministic
Cons: Can be repetitive, low creativity

---

### **7.2 Beam Search**

Explore multiple beams (candidate sequences) simultaneously.

Pros: Better quality
Cons: Slower

---

### **7.3 Sampling-Based Methods**

Used mostly in LLMs for creative generation:

* Top-k sampling
* Top-p (nucleus) sampling
* Temperature scaling
* Repetition penalties

Pros: Diversity, human-like text
Cons: Stochastic, less stable

---

## **8. Parallelism and Optimization Techniques in Inference**

Modern LLM inference involves heavy optimizations:

### **8.1 KV Caching**

Reduces quadratic attention cost.

### **8.2 FlashAttention / Memory-Efficient Attention**

Optimizes softmax attention.

### **8.3 Quantization**

Converts weights to lower precision (INT8, INT4, FP8).

### **8.4 Speculative Decoding**

Small model drafts output, large model verifies.

### **8.5 Tensor Parallelism and Pipeline Parallelism**

Used in large multi-GPU inference.

### **8.6 Continuous Batching**

Serve multiple user requests at once by merging attention computations.

---

## **9. End-to-End Example (Simplified)**

Given prompt:
**“The capital of France is”**

### Step 1: Tokenize

`[1012, 99, 4021, 19, 17, ...]`

### Step 2: Generate next token

Model outputs probabilities:

* Paris: 0.92
* Lyon: 0.03
* …

### Step 3: Final output

“Paris”

This process repeats for long responses.

---

## **10. Summary and Key Takeaways**

* **Encoder-only models** run in parallel → fast inference.
* **Decoder-only models** generate tokens one at a time → slow but powerful.
* Transformers use **cached K/V** to avoid recomputing attention for the whole sequence.
* Inference heavily relies on **auto-regression**, **causal masking**, and **generation strategies**.
* Optimizations such as **quantization**, **flash attention**, **tensor parallelism**, and **speculative decoding** make LLM inference practical.

---

## **11. Optional Add-on: Ready-Made Section Titles for Documents**

You can expand your document using these headers:

* Introduction to Transformer Architecture
* Training vs. Inference
* Role of Self-Attention in Inference
* KV Caching Explained Visually
* Efficient Transformer Inference for Large Models
* Inference Challenges in Long-Context Models
* Real-Time Inference and Serving Considerations
* Hardware Acceleration for Transformer Inference
* Conclusion

---
