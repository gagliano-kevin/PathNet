#==============================================================================================================================================================
#==============================================================================================================================================================
#-------------------------------------------------- python -m tests.scaling.random_parameter_scaling ------------------------------------------------------
#==============================================================================================================================================================
#==============================================================================================================================================================

import torch
import torch.nn as nn
import numpy as np
import tracemalloc

from source.PathNet import Trainer, TrainerLayerWiseKernel, TrainerRandomSampling
from source.utils.dataset_utils.sine_utils import generate_sinusoidal_tensor
from source.utils.plot_utils import save_metrics, load_metrics, plot_individual_algorithms, plot_all_algorithms

TEST_NAME = "random_parameter_scaling"

ITERATIONS = 50

MODEL_NAME_PREFIX = "sine_model"
DATASET_NAME = "Noisy Sine Function"

SAVE_TRAINED_MODEL = False
DELTA_ABS = None
EARLY_STOPPING = False

E_S_PATIENCE = 200
LOSS_IMPROVEMENT_THRESHOLD = 1e-3
PARAMETER_RANGE = (-10, 10)
QUANTIZATION_FACTOR = 10
BEAM_WIDTH = 1e3

# Parameters for synthetic Sine Dataset
NUM_SAMPLES = 50
MIN_ANGLE = 0
MAX_ANGLE = 2 * np.pi
NOISE_LEVEL = 0.1

INPUT_SIZE = 1
OUTPUT_SIZE = 1

labels_list = ["A-star Single Kernel", "A-star Layer-Wise Kernels", "A-star Random Sampling"]

# =========================================================================================================================================================
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------- SMALL NET --------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------
# =========================================================================================================================================================

# Architectures to test: List of tuples (HIDDEN_SIZE_1, HIDDEN_SIZE_2)
# Scaling up the hidden dimensions will smoothly scale the total parameter count
NETWORK_ARCHITECTURES = [
    (16, 16),
    (32, 16),
    (32, 32),
    (64, 32),
    (64, 64),
    (128, 64),
    (128, 128),
    (256, 128)
]

# Parameters for Random Sampling Neighbors Generation
PERTURBATION_RATIO = 0.01       
SEARCH_COVERAGE_RATIO = 0.01     

metrics_list = [
    {
        "training_times": [],
        "number_of_parameters": [],
        "memory_usage_mb": [],
    } for _ in range(len(labels_list))
]

# Helper function to count parameters
count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)

# Generate dataset once for all models to ensure identical training conditions
X_train, Y_train = generate_sinusoidal_tensor(num_samples=NUM_SAMPLES, min_angle=MIN_ANGLE, max_angle=MAX_ANGLE, noise_level=NOISE_LEVEL)
print(f"\nTraining Data Shape: {X_train.shape}, {Y_train.shape}")

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------- RANDOM SAMPLING NEIGHBORS GENERATION -------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for hidden_1, hidden_2 in NETWORK_ARCHITECTURES:

    print(f"\n--- TEST NAME: {TEST_NAME} \t Random Sampling \t Hidden Sizes: ({hidden_1}, {hidden_2}) ---\n")

    model = nn.Sequential(
            nn.Linear(INPUT_SIZE, hidden_1),  
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, OUTPUT_SIZE),
            )
            
    num_params = count_parameters(model)

    trainer = TrainerRandomSampling(model=model,
                            loss_fn=nn.MSELoss(),
                            quantization_factor=QUANTIZATION_FACTOR,
                            parameter_range=PARAMETER_RANGE,
                            debug_mlp=False,
                            perturbation_ratio=PERTURBATION_RATIO, search_coverage_ratio=SEARCH_COVERAGE_RATIO,
                            delta_abs=DELTA_ABS,
                            early_stopping=EARLY_STOPPING, e_s_patience=E_S_PATIENCE,
                            dynamic_quantization=False,
                            loss_improvement_threshold=LOSS_IMPROVEMENT_THRESHOLD,
                            max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=SAVE_TRAINED_MODEL, model_name=MODEL_NAME_PREFIX + f'_random_sampling_astar_run'
                            )

    # Start tracing memory
    tracemalloc.start()

    trainer.beam_search_opt_train(X_train, Y_train, BEAM_WIDTH)

    # Capture peak memory and stop tracing
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Convert bytes to Megabytes and save
    peak_mem_mb = peak_mem / (1024 * 1024)

    metrics_list[2]["training_times"].append(trainer.training_time)
    metrics_list[2]["number_of_parameters"].append(num_params)
    metrics_list[2]["memory_usage_mb"].append(peak_mem_mb)

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- COMPARISON ---------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

all_results = {label: metric for label, metric in zip(labels_list, metrics_list)}

save_metrics(all_results, TEST_NAME)

metrics = load_metrics(TEST_NAME)

# --- TIME PLOTS ---
plot_individual_algorithms(
    metrics, 
    prefix=TEST_NAME + "_time", 
    directory=TEST_NAME, 
    x_key="number_of_parameters", x_label="Number of Model Parameters",
    y_key="training_times", y_label="Training Time (s)"
)

plot_all_algorithms(
    metrics, 
    output_filename=TEST_NAME + "_time_comparison.png", 
    directory=TEST_NAME,
    x_key="number_of_parameters", x_label="Number of Model Parameters",
    y_key="training_times", y_label="Training Time (s)"
)

# --- MEMORY PLOTS ---
plot_individual_algorithms(
    metrics, 
    prefix=TEST_NAME + "_memory", 
    directory=TEST_NAME, 
    x_key="number_of_parameters", x_label="Number of Model Parameters",
    y_key="memory_usage_mb", y_label="Peak Memory Usage (MB)"
)

plot_all_algorithms(
    metrics, 
    output_filename=TEST_NAME + "_memory_comparison.png", 
    directory=TEST_NAME,
    x_key="number_of_parameters", x_label="Number of Model Parameters",
    y_key="memory_usage_mb", y_label="Peak Memory Usage (MB)"
)
