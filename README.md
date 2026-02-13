# PathNet: High-Precision Neural Network Training via A* Search

**PathNet** is a research framework that reframes neural network training as a **state-space exploration problem**. Instead of relying on gradient descent (backpropagation), PathNet utilizes **A* Search** and **Beam Search** to navigate a discrete, quantized weight space.

By treating weight updates as transitions in a graph, PathNet prioritizes **numerical precision, stability, and interpretability** over raw speed. It is particularly robust for non-differentiable objectives, escaping local minima, and ensuring deterministic convergence properties.

---

## ✨ Key Features

### 🧠 Core Search Algorithms

* **Quantized MLP**: Wraps standard PyTorch models, discretizing parameters into a searchable grid defined by a `quantization_factor` and `parameter_range`.
* **Beam Search A***: Implements `beam_search_opt_train` to explore the weight space while pruning high-cost nodes, preventing memory saturation during deep searches.
* **Memory Guard**: Integrated `SystemMemoryGuard` to gracefully terminate training if system RAM usage exceeds safety thresholds.

### 🔍 Versatile Neighbor Generation

PathNet supports three distinct strategies for exploring the weight space, handled by specialized Trainer classes:

1. **Single-Kernel (`Trainer`)**: Applies a uniform kernel (K × K) and stride across all layers. Best for homogeneous architectures.
2. **Layer-Wise Kernels (`TrainerLayerWiseKernel`)**: Allows defining specific kernel sizes and strides for *each* layer (e.g., broad search in early layers, fine-tuning in later layers).
3. **Random Sampling (`TrainerRandomSampling`)**: Perturbs a random ratio of parameters. Useful for high-dimensional spaces where structured kernels are too computationally expensive.

### ⚡ Dynamic Optimization Heuristics

To escape local minima and refine precision on the fly, PathNet implements adaptive mechanisms:

* **Dynamic Kernel Reshaping**: Automatically shrinks kernel sizes and strides (e.g., `4x4` → `3x3`) when training patience is exhausted (`d_k_r_patience`).
* **Dynamic Quantization**: Multiplies the `quantization_factor` (e.g., 10 → 100) upon stagnation, instantly increasing the numerical resolution of the search grid when `d_q_patience` is reached.

---

## 📂 Project Structure

```text
PATHNET/
├── PathNet_report/            # Documentation and reports
├── results/                   # Training results and outputs
├── source/                    # Main source code directory
│   |
│   └── utils/                 # Utility modules
│       |
│       ├── dataset_utils/     # Dataset loading and preprocessing
│       │   |
│       │   ├── circle_utils.py
│       │   ├── housing_utils.py
│       │   ├── iris_utils.py
│       │   ├── sine_utils.py
│       │   └── wine_utils.py
│       ├── adjust_plots.py    # Script to regenerate plots from json file
│       ├── evaluation_utils.py # Metrics and model evaluation
│       ├── memory_guard.py    # System RAM monitoring
│       ├── models.py          # Neural network architectures
│       ├── neighbors_utils.py # Neighbor generation for search
│       └── plot_utils.py      # Additional plotting functions
├── __init__.py
├── PathNet.py                 # Core PathNet implementation
├── tests/                     # Test and benchmark scripts
│   ├── adam_vs_astar/         # Comparison experiments
│   │   |
│   │   ├── california_housing.py
│   │   ├── iris.py
│   │   ├── sine.py
│   │   └── wine.py
│   └── neighbors_generation_strategies/  # Strategy comparison tests
│       |
│       ├── comparison_with_beam_search/
│       │   └── california_housing.py
│       ├── comparison_with_no_beam_search/
│       │   └── california_housing.py
│       ├── grid_search_random_sampling/
│       │   └── california_housing.py
│       ├── static_vs_dynamic/
│       └── vanilla_train_vs_beam_search/
│           └── california_housing.py
├── .gitignore
├── mlp_astar.yml              # Environment configuration
└── README.md                  # Project documentation
```

---

## ▶️ Usage

PathNet provides specific `Trainer` classes for different search strategies. Below are examples based on the California Housing regression task.

### 1. Single-Kernel Beam Search

Use `Trainer` for applying a uniform search window across the entire network.

```python
import torch.nn as nn
from source.PathNet import Trainer

# Define Model
model = nn.Sequential(
    nn.Linear(8, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    loss_fn=nn.MSELoss(),
    quantization_factor=10,
    parameter_range=(-10, 10),
    # Search Strategy
    weight_kernel=[2, 2], 
    bias_kernel=[2],
    x_stride=1, 
    y_stride=1,
    # Dynamic Heuristics
    dynamic_quantization=True, 
    d_q_patience=100,
    dynamic_kernel_reshaping=True, 
    d_k_r_patience=100,
    max_iterations=1000
)

# Train using Beam Search
trainer.beam_search_opt_train(X_train, Y_train, beam_width=500)
```

### 2. Layer-Wise Kernel Search

Use `TrainerLayerWiseKernel` to customize the search granularity per layer.

```python
from source.PathNet import TrainerLayerWiseKernel

# Define specific kernels for a 3-layer network
weight_kernels = [[4, 4], [2, 2], [1, 2]]  # Layer 1, 2, 3
bias_kernels = [[2], [1], [1]]
weight_strides = [[2, 2], [1, 1], [1, 1]]
bias_strides = [[1], [1], [1]]

trainer = TrainerLayerWiseKernel(
    model=model,
    loss_fn=nn.MSELoss(),
    quantization_factor=10,
    parameter_range=(-10, 10),
    weight_kernels=weight_kernels,
    bias_kernels=bias_kernels,
    weight_strides=weight_strides,
    bias_strides=bias_strides,
    max_iterations=1000
)

trainer.beam_search_opt_train(X_train, Y_train, beam_width=500)
```

### 3. Random Sampling Search

Use `TrainerRandomSampling` for stochastic exploration in large parameter spaces.

```python
from source.PathNet import TrainerRandomSampling

trainer = TrainerRandomSampling(
    model=model,
    loss_fn=nn.MSELoss(),
    quantization_factor=10,
    parameter_range=(-10, 10),
    perturbation_ratio=0.01,       # Perturb 1% of weights per neighbor
    search_coverage_ratio=0.1,     # Generate neighbors = 10% of total params
    max_iterations=1000
)

trainer.beam_search_opt_train(X_train, Y_train, beam_width=500)
```

---

## ⚙️ Configuration Parameters

| Parameter | Description |
| :--- | :--- |
| **`quantization_factor`** | Defines the grid resolution (step size = 1/QF). Higher values provide more precision but create a larger search space. |
| **`beam_width`** | Controls the breadth of the search. Keeps only the top-N most promising nodes per iteration. |
| **`loss_improvement_threshold`** | Minimum loss decrease required to reset patience counters. |
| **`dynamic_quantization`** | If `True`, multiplies `quantization_factor` by `quantization_factor_multiplier` after `d_q_patience` iterations of stagnation. |
| **`dynamic_kernel_reshaping`** | If `True`, reduces kernel/stride sizes by `_decr` values after `d_k_r_patience` iterations of stagnation. |
| **`early_stopping`** | Terminates training if no improvement is seen after `e_s_patience` iterations. |

---

## 📊 Benchmarks

PathNet includes comprehensive benchmark scripts to evaluate performance against traditional optimizers and compare different search strategies.

### A* vs Adam Optimizer Comparisons

Compare PathNet's A* search against the Adam optimizer on various datasets:

```bash
# California Housing dataset
python -m tests.adam_vs_astar.california_housing

# Iris classification dataset
python -m tests.adam_vs_astar.iris

# Sine wave regression
python -m tests.adam_vs_astar.sine

# Wine classification dataset
python -m tests.adam_vs_astar.wine
```

### Neighbor Generation Strategy Comparisons

Evaluate different search strategies and configurations:

**Beam Search vs No Beam Search:**
```bash
python -m tests.neighbors_generation_strategies.comparison_with_beam_search.california_housing
python -m tests.neighbors_generation_strategies.comparison_with_no_beam_search.california_housing
```

**Grid Search vs Random Sampling:**
```bash
python -m tests.neighbors_generation_strategies.grid_search_random_sampling.california_housing
```

**Vanilla Training vs Beam Search:**
```bash
python -m tests.neighbors_generation_strategies.vanilla_train_vs_beam_search.california_housing
```

### Benchmark Results

Results and training logs are automatically saved to the `results/` directory. Output files include:
- Training loss curves
- Convergence statistics
- Comparison metrics (MSE, accuracy, training time)
- Model checkpoints

Each experiment folder may contain `*_output.txt` files with detailed performance metrics.

---

## 🚀 Future Extensions

- [ ] **Multi-GPU Parallelism:** Distributing the `Open Set` evaluation across multiple GPUs.
- [ ] **Hybrid Optimization:** A "hand-off" protocol that starts with Adam for speed and switches to A* for high-precision refinement.
- [ ] **CNN Support:** Extending 2D kernels to 3D filter volumes for computer vision tasks.

---

## 📝 Citation

If you use PathNet in your research, please cite:

```bibtex
@software{PathNet2026,
  title={PathNet: A* Search for Quantized Neural Network Training},
  author={Kevin Gagliano},
  year={2026},
  url={https://github.com/gagliano-kevin/PathNet},
  note={GitHub repository}
}
```

---

## 📄 License

This project is open-source and available for research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue on the GitHub repository.
