# Day_039 | 🔧 Keras Tuner: Hyperparameter Tuning

**Keras Tuner** is a Python library that helps you find the optimal set of **hyperparameters** for your TensorFlow/Keras models. It provides infrastructure to systematically search a defined space of hyperparameter values, automating the process that is typically done manually or with ad-hoc scripts.

It works by integrating directly with your Keras model definition, allowing you to easily define a "search space" and execute various search algorithms.

-----

### 1\. Key Components of Keras Tuner

To use Keras Tuner, you primarily interact with three components:

| Component | Description |
| :--- | :--- |
| **HyperModel** | A function or subclass that **defines the model creation process**. It takes a `hp` object (the hyperparameter instance) and returns a compiled Keras model. This is where you define the search space. |
| **Hyperparameters (`hp`)** | The object used within the HyperModel to define the range or choices for parameters like learning rate, number of layers, or activation functions. |
| **Tuner** | The search algorithm that drives the optimization. It manages the process of building, training, and evaluating models for different hyperparameter combinations. |

-----

### 2\. Defining the Search Space (Hyperparameters)

Inside your `HyperModel` function, you define the parameters you want to tune using the following methods on the `hp` object:

| Method | Use Case | Example |
| :--- | :--- | :--- |
| **`hp.Choice(name, values)`** | Selects one value from a discrete list of options. | `hp.Choice('activation', ['relu', 'tanh'])` |
| **`hp.Int(name, min, max, step)`** | Selects an integer value within a range. | `hp.Int('units', min_value=32, max_value=512, step=32)` |
| **`hp.Float(name, min, max, sampling)`** | Selects a floating-point value within a range. | `hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')` |
| **`hp.Boolean(name)`** | Selects either True or False. | `hp.Boolean('batch_norm')` |

### 3\. Common Tuner Algorithms

Keras Tuner offers different algorithms to navigate the hyperparameter space efficiently:

| Tuner | Mechanism | Advantage |
| :--- | :--- | :--- |
| **RandomSearch** | Tries random combinations of hyperparameters within the defined space. | Simple, fast, and often effective for a first pass, as not all hyperparameters are equally important. |
| **Hyperband** | A bandit-based approach that allocates resources adaptively. It trains many models for a short time and prunes the worst-performing ones, focusing resources on the most promising candidates. | Highly efficient, especially for large search spaces and limited computational resources. |
| **BayesianOptimization** | Uses a statistical model (Gaussian process) to model the objective function (e.g., validation loss). It uses this model to intelligently select the next best hyperparameter combination to try. | Generally finds better optima with fewer total trials, but each step is computationally slower. |

-----

### 4\. General Workflow

The typical usage of Keras Tuner follows these steps:

1.  **Define the HyperModel:** Create a function (e.g., `build_model(hp)`) that includes the search space definitions.
2.  **Instantiate the Tuner:** Choose a search algorithm (e.g., `Hyperband`) and point it to your `build_model` function.
    
    > Python
    ```python
    tuner = kt.Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=10,
        directory='my_dir',
        project_name='intro_keras_tuner'
    )
    ```
3.  **Start the Search:** Call the `search` method, passing in your training data, similar to calling `model.fit()`.
4.  **Retrieve the Best Model:** After the search is complete, get the optimal set of hyperparameters and build the final, best-performing model.

    > Python
    ```python
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hps)
    ```

Here’s a clear, practical guide to **Keras Tuner** and how to use it for **hyperparameter tuning**.

---

## 🔧 **Keras Tuner — Hyperparameter Tuning Guide**

**Keras Tuner** is a library that automatically searches for the best hyperparameters for your Keras/TensorFlow models.
It supports several search algorithms:

* **RandomSearch**
* **Hyperband**
* **BayesianOptimization**
* **SklearnTuner (for scikit-learn models)**

---

## 🚀 1. **Installation**

```bash
pip install keras-tuner
```

---

## 🧠 2. Basic Workflow

1. **Define a model-building function (`build_model`)**
   – It receives a `HyperParameters` object (`hp`)
   – You define tunable values inside the function

2. **Create a tuner (RandomSearch, Hyperband, etc.)**

3. **Run the search**

4. **Get the best model/hyperparameters**

---

## 📌 3. Example: Tuning a Dense Neural Network

### **Step 1: Define the model**

> Keras

```python
import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras import layers

def build_model(hp):
    model = keras.Sequential()
    
    # Tune number of layers
    for i in range(hp.Int("num_layers", 1, 4)):
        model.add(
            layers.Dense(
                units=hp.Int(f"units_{i}", min_value=32, max_value=256, step=32),
                activation=hp.Choice("activation", ["relu", "tanh"])
            )
        )
    
    model.add(layers.Dense(10, activation="softmax"))

    # Tune learning rate
    lr = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
    
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
```

---

## 🚦 Step 2: Choose a Tuner

### **Random Search**

```python
tuner = kt.RandomSearch(
    build_model,
    objective="val_accuracy",
    max_trials=20,
    executions_per_trial=1,
    directory="my_tuning",
    project_name="random_search"
)
```

### **Hyperband (faster for large models)**

```python
tuner = kt.Hyperband(
    build_model,
    objective="val_accuracy",
    max_epochs=20,
    factor=3,
    directory="my_tuning",
    project_name="hyperband"
)
```

### **Bayesian Optimization**

```python
tuner = kt.BayesianOptimization(
    build_model,
    objective='val_accuracy',
    max_trials=20,
    directory='my_tuning',
    project_name='bayesian'
)
```

---

## 📌 Step 3: Start the Search

```python
tuner.search(x_train, y_train,
             epochs=10,
             validation_split=0.2)
```

---

## ⭐ Step 4: Get the Best Model

```python
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(best_hps.values)

best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()
```

---

## 🧪 Step 5: Train the Best Model

```python
history = best_model.fit(
    x_train, y_train,
    epochs=20,
    validation_split=0.2
)
```

---

## 🔍 Useful Hyperparameter Types

| Type         | Description      | Examples                |
| ------------ | ---------------- | ----------------------- |
| `hp.Int`     | Integer range    | units, layers           |
| `hp.Float`   | Float range      | learning rate           |
| `hp.Choice`  | Choose from list | activation fn           |
| `hp.Boolean` | True/False       | batch normalization     |
| `hp.Fixed`   | Fixed value      | when tuning selectively |

---

## 📘 Example: Tuning CNN Hyperparams (Short)

> Python
```python
def build_cnn(hp):
    model = keras.Sequential()
    model.add(
        layers.Conv2D(
            filters=hp.Int("filters", 32, 128, step=32),
            kernel_size=hp.Choice("kernel_size", [3, 5]),
            activation="relu",
            input_shape=(28, 28, 1)
        )
    )
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation="softmax"))

    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Float("lr", 1e-4, 1e-2, sampling="log")
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
```

---

## 🏁 Summary

Keras Tuner helps you find the best model by exploring:

* number of layers
* neurons per layer
* activation functions
* optimizers
* learning rates
* batch size
* regularization

…in a clean, scalable way.

---
