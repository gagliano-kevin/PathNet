# PathNet: Quantized MLP Training with A* Search

This repository implements **PathNet**, a framework for training quantized multi-layer perceptrons (MLPs) using **A* search** instead of gradient descent.
The examples provided is focused on solving classification and regression tasks on simple/small datasets, but the design can be extended to more complex tasks.

The key idea is to treat model training as a search problem in a quantized weight space. A* explores weight configurations to minimize the loss function.

---

## ✨ Features

* **Quantized MLP**:
  All parameters are discretized within a configurable range using a quantization factor.

* **A* Training**:
  Instead of gradient descent, the training algorithm uses A* search to find optimal weights.

* **Overflow Handling**:
  Parameters that exceed the allowed range are clipped or discarded.

* **Logging & Visualization**:

  * Save training logs to text files.
  * Plot loss curves (`loss`, `g`, `f`) over iterations.

* **Grid Search**:
  Run experiments across multiple hyperparameter combinations automatically.

* **Model Saving/Loading**:
  Store and reload trained models with quantization support.

---

## 📂 Project Structure

* `QuantizedMLP`:
  Wrapper for an MLP with quantization, evaluation, and hashing capabilities.

* `SearchNode`:
  Node representation for A* search (with `g`, `h`, `f` values).

* `Trainer`:
  Trains a quantized MLP using A* search, manages training loop, plotting, logging, and saving models.

* `GridSearchTrainer`:
  Automates experiments over multiple hyperparameter combinations.


---

## ▶️ Usage

### 1. Train a Single Model

Example (training on XOR):

```python
from pathnet import Trainer, XOR_DATASET, XOR_LABELS
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(2, 2),
    nn.ReLU(),
    nn.Linear(2, 1),
    nn.Sigmoid()
)

trainer = Trainer(
    model,
    nn.MSELoss(),
    quantization_factor=2,
    parameter_range=(-4, 4),
    max_iterations=2000,
    target_loss=0.0001,
)

trainer.train(XOR_DATASET, XOR_LABELS)
trainer.plot("xor_loss.png")
trainer.save_model("xor_model.pth")
```

### 2. Run Grid Search

```python
from pathnet import GridSearchTrainer, XOR_DATASET, XOR_LABELS
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(2, 2),
    nn.ReLU(),
    nn.Linear(2, 1),
    nn.Sigmoid()
)

grid_search_trainer = GridSearchTrainer(
    models=[model],
    loss_funcs=[nn.MSELoss()],
    quantization_factors=[1, 2],
    parameter_ranges=[(-5, 5)],
    param_fractions=[1.0],
    max_iterations=[1000],
    log_freq=[500],
    target_losses=[0.0001],
    debug_mlps=True
)

grid_search_trainer.run_grid_search(XOR_DATASET, XOR_LABELS, log_filename="grid_search_results.txt")
```

---

## 📊 Outputs

* **Logs**:
  Training logs saved in `.txt` files with iteration-wise metrics.

* **Plots**:
  PNG plots of loss (`h`), cost (`g`), and total cost (`f`).

* **Models**:
  Best-performing models saved in `.pth` files.

---

## 🔧 Key Parameters

* `quantization_factor`: Controls granularity of parameter values (higher = finer).
* `parameter_range`: Allowed weight range (tuple).
* `param_fraction`: Fraction of weights perturbed per neighbor generation.
* `max_iterations`: Maximum iterations for A*.
* `target_loss`: Early stopping threshold.

---

## 🚀 Future Extensions

* Support for larger datasets (MNIST, etc.)
* Parallelized search across nodes
* Custom heuristics for `h` in A*

---