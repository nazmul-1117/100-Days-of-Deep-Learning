# Day_072 | What is Self Attention

## 💡 What is Self-Attention?

**Self-Attention** (or intra-attention) is the core mechanism of the **Transformer** architecture. It is a technique that allows a model to weigh the importance of all other elements within a single input sequence relative to the current element being processed.

In simpler terms, when a model is processing a word, Self-Attention helps it look at the entire sentence and decide which other words are most relevant for understanding the current word's context and meaning.

### Example: Resolving Ambiguity

Consider the sentence: "The **animal** didn't cross the road because **it** was too wide."

When processing the word "**it**," a Self-Attention mechanism will assign a high weight (high attention score) to the word "**road**" and a low weight to words like "**animal**." This tells the model that "it" refers to the "road."

---

## 🛑 Why We Need Self-Attention Over Traditional Embeddings

Self-Attention was developed to solve the critical limitations of previous word representations and context modeling techniques.

### 1. Traditional Count-Based Embeddings

Traditional models like **One-Hot Encoding (OHE)**, **Bag-of-Words (BoW)**, and **TF-IDF** suffer from two major problems:

| Method | Problem | Why Self-Attention is Better |
| :--- | :--- | :--- |
| **OHE, BoW, TF-IDF** | **Loss of Word Order/Context:** These methods treat text as an unordered collection of words. They cannot capture grammar, syntax, or the sequential meaning of a sentence. | Self-Attention directly models the relationship between every pair of words in the exact order they appear, preserving all context. |
| **OHE, BoW, TF-IDF** | **Sparsity and High Dimensionality:** The vectors are very large (equal to the vocabulary size) and mostly filled with zeros (sparse). | Self-Attention works on dense, low-dimensional **embeddings** which are computationally efficient. |

### 2. Problems with Static Word Embeddings

**Word Embeddings** (like **Word2Vec** or **GloVe**) are a significant improvement, representing each word as a dense vector. However, they introduce the problem of **Ambiguity/Polysemy**:

| Method | Problem | Why Self-Attention is Better |
| :--- | :--- | :--- |
| **Word Embeddings (Word2Vec, GloVe)** | **Static Context (Monosemy):** A single word (e.g., "bank") always has the exact same vector, regardless of its context in the sentence (river bank vs. financial bank). | Self-Attention generates **Contextualized Embeddings**. The representation of "bank" changes based on the other words in that specific sentence ("river" or "loan"), resolving polysemy. |

### 3. Problems with RNN/LSTM Context Modeling

Recurrent models (RNNs/LSTMs) capture context, but they suffer from fundamental efficiency issues:

| Method | Problem | Why Self-Attention is Better |
| :--- | :--- | :--- |
| **RNNs/LSTMs** | **Sequential Dependency/Slow Training:** They must process the sequence one word at a time, making training slow and impossible to fully parallelize. | Self-Attention computes relationships between **all words simultaneously**, enabling massive parallel processing and dramatically faster training. |
| **RNNs/LSTMs** | **Long-Term Dependency:** Despite LSTMs, they still struggle to connect words separated by many steps due to the inherent sequential path. | The path between any two words in a Transformer is constant (short), allowing it to easily model **very long-range dependencies**. |

**In summary, Self-Attention provides the best of all worlds: it creates dense, low-dimensional representations that are fully contextualized, and it does so in a way that allows for highly efficient, parallel computation.**


Below is a clean, structured **Part 1** explanation of **Self-Attention** focusing on:
**(1) What it is**, **(2) Why we need it**, **(3) Why older feature/embedding methods (OHE, BoW, TF-IDF, Word Embeddings) fall short**, and **(4) What problems self-attention solves.**

---

## **1. What Is Self-Attention?**

**Self-attention** is a mechanism that allows a model to **look at every other token in a sequence and decide which ones matter most** when understanding the current token.

In simple terms:

> Self-attention calculates **how much each word should pay attention to all other words** in the sentence.

Example:
In the sentence *“The cat that chased the mouse was hungry”*
the model needs to connect **“cat” ↔ “was hungry”**, even though they are far apart.
Self-attention creates these connections automatically.

---

## **2. Why Do We Need Self-Attention Instead of Older Embeddings?**

Older text-representation methods (OHE, BoW, TF-IDF, Word2Vec, GloVe) **cannot capture context and relationships effectively**.
Self-attention solves their limitations.

Let’s break it down.

---

## **3. Problems With Older Embedding Methods**

## **A. One-Hot Encoding (OHE)**

**What it does:**
Represents each word as a vector with a single “1” and the rest “0”.

**Problems:**

* No meaning: “cat” and “dog” are equally distant (completely unrelated).
* Extremely sparse and high-dimensional.
* Vocabulary-dependent (huge vectors).
* Cannot capture *any* context.

➡️ Useless for understanding relationships or sentence-level meaning.

---

## **B. Bag-of-Words (BoW)**

**What it does:**
Just counts how many times each word appears.

**Problems:**

* **No word order**
  “Dog bites man” = “Man bites dog”
* **No context**
  It treats all appearances of a word the same.
* Very high dimensional.
* Cannot capture meaning or relationships.

➡️ Totally ignores structure and semantics.

---

## **C. TF–IDF**

**What it does:**
Weighted BoW (downweights common words).

**Problems:**

* Still ignores order.
* Still ignores meaning.
* Still cannot capture relationships.
* Different contexts of a word look the same.

➡️ Slightly smarter BoW, but still no real understanding.

---

## **D. Word Embeddings (Word2Vec, GloVe)**

These are dense, low-dimensional embeddings where words with similar meanings are close.

**Strengths:**

* Capture semantic similarity
  (king ≈ queen, dog ≈ puppy)

**BUT serious problems:**

### **1. Static embeddings**

Each word has **only one meaning**.

Example:

* “bank” (river bank)
* “bank” (money bank)
  → both get the same vector.

This kills contextual understanding.

### **2. Cannot capture relationships in a sentence**

Word embeddings don’t change based on:

* grammar
* long-range dependencies
* roles (subject, object)

### **3. No way to model which words depend on which**

E.g., in
“John went to the store because **he** was hungry”
Word embeddings cannot determine “he” → “John”.

### **4. No positional awareness**

Embeddings don’t know:

* who modifies what
* which words belong together
* which words are far vs near

➡️ Word embeddings are **context-blind**.

---

## **4. Why Self-Attention Solves These Problems**

Self-attention fixes all limitations above.

### **✓ 1. Contextual meaning**

The same word gets **different vectors depending on surrounding words**.

“bank” in

* “river bank”
* “bank account”
  becomes two different representations.

### **✓ 2. Captures relationships between any two words**

Regardless of their distance in the sentence.

Example:
In

> “The book that you gave me was excellent.”
> Self-attention knows that:

* “book” ↔ “was excellent”
* “you” ↔ “gave”

### **✓ 3. Global view of the entire sequence**

Unlike RNNs or CNNs, which have limited range,
self-attention attends to **all tokens at once**.

### **✓ 4. Learns what is important**

Automatically assigns weights (attention scores):

* High → strong relationship
* Low → weak relationship

### **✓ 5. Produces contextual embeddings**

Self-attention transforms static word embeddings into rich, dynamic, contextualized versions.

---

## **Summary Table**

| Method             | Understands Meaning? | Uses Word Order? | Context-Aware? | Learns Relationships? |
| ------------------ | -------------------- | ---------------- | -------------- | --------------------- |
| One-Hot            | ❌                    | ❌                | ❌              | ❌                     |
| BoW                | ❌                    | ❌                | ❌              | ❌                     |
| TF-IDF             | ❌                    | ❌                | ❌              | ❌                     |
| Word2Vec/GloVe     | ✔️                   | ❌                | ❌              | ❌                     |
| **Self-Attention** | **✔️✔️**             | **✔️**           | **✔️✔️**       | **✔️✔️✔️**            |

Self-attention is the **first method** that:

* understands meaning **in context**
* understands sentence structure
* understands long-range dependencies
* scales to modern deep learning (Transformers)

---