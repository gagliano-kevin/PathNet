#==============================================================================================================================================================
#==============================================================================================================================================================
#-------------------------------------------------- python -m tests.scaling.data_scaling ----------------------------------------------------------
#==============================================================================================================================================================
#==============================================================================================================================================================

import torch
import torch.nn as nn
import numpy as np
import tracemalloc

from source.PathNet import Trainer, TrainerLayerWiseKernel, TrainerRandomSampling
from source.utils.dataset_utils.sine_utils import generate_sinusoidal_tensor
from source.utils.plot_utils import save_metrics, load_metrics, plot_individual_algorithms, plot_all_algorithms

ITERATIONS = 1000

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
TRAINING_SAMPLES = [10, 50, 100, 500, 1000, 5000, 10000]
MIN_ANGLE = 0
MAX_ANGLE = 2 * np.pi
NOISE_LEVEL = 0.1

INPUT_SIZE = 1
OUTPUT_SIZE = 1

EARLY_STOPPING = False
E_S_PATIENCE = 200
LOSS_IMPROVEMENT_THRESHOLD = 1e-3
PARAMETER_RANGE = (-10, 10)


labels_list = ["A-star Single Kernel", "A-star Layer-Wise Kernels", "A-star Random Sampling"]


# =========================================================================================================================================================
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------- SMALL NET --------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------
# =========================================================================================================================================================

TEST_NAME = "small_net_sine_data_scaling"

# Neural Network Settings
HIDDEN_SIZE_1 = 32
HIDDEN_SIZE_2 = 16

# Parameter for single Kernel Neighbors Generation
WEIGHT_KERNEL = [2,2]
BIAS_KERNEL = [2]
X_STRIDE = 1
Y_STRIDE = 1

# Parameters for Layer-Wise Kernels Neighbors Generation
WEIGHT_KERNELS = [[2,2], [2,2], [1,2]]
BIAS_KERNELS = [[2], [2], [1]]
WEIGHT_STRIDES = [[1,1], [1,1], [1,1]]      # Format: list of [x_stride, y_stride] per layer
BIAS_STRIDES = [[1], [1], [1]]              # Format: list of [stride] per layer

# Parameters for Random Sampling Neighbors Generation
PERTURBATION_RATIO = 0.01       # 1% of the parameters will be perturbed per each neighbor
SEARCH_COVERAGE_RATIO = 0.1     # 10% of the total number of parameters in the model will be the number of neighbors generated per each state

metrics_list = [
    {
        "training_times": [],
        "number_of_training_samples": [],
        "memory_usage_mb": [],
    } for _ in range(len(labels_list))
]

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------- SINGLE KERNEL NEIGHBORS GENERATION ------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for num_samples in TRAINING_SAMPLES:
    
    X_train, Y_train = generate_sinusoidal_tensor(num_samples=num_samples, min_angle=MIN_ANGLE, max_angle=MAX_ANGLE, noise_level=NOISE_LEVEL)

    print(f"\nTraining Data Shape: {X_train.shape}, {Y_train.shape}")

    print(f"\n--- TEST NAME: {TEST_NAME} \t Single Kernel Neighbors Generation \t BEAM SEARCH ASTAR Training ---\n")

    model = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE_1),  
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_2, OUTPUT_SIZE),
            )


    trainer = Trainer(model=model,
                            loss_fn=nn.MSELoss(),
                            quantization_factor=QUANTIZATION_FACTOR,
                            parameter_range=PARAMETER_RANGE,
                            debug_mlp=False,
                            #----------------------------------------------------------------------------------
                            weight_kernel = WEIGHT_KERNEL, bias_kernel = BIAS_KERNEL, x_stride=X_STRIDE, y_stride=Y_STRIDE, delta_abs=DELTA_ABS,
                            #----------------------------------------------------------------------------------
                            early_stopping=EARLY_STOPPING, e_s_patience=E_S_PATIENCE,
                            #----------------------------------------------------------------------------------
                            dynamic_quantization=False,
                            #-----------------------------------------------------------------------------------
                            dynamic_kernel_reshaping=False,
                            #----------------------------------------------------------------------------------
                            loss_improvement_threshold=LOSS_IMPROVEMENT_THRESHOLD,
                            #----------------------------------------------------------------------------------
                            max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=SAVE_TRAINED_MODEL, model_name=MODEL_NAME_PREFIX + f'_single_kernel_astar'
                            )

    # Start tracing memory
    tracemalloc.start()
    
    trainer.beam_search_opt_train(X_train, Y_train, BEAM_WIDTH)
    
    # Capture peak memory and stop tracing
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Convert bytes to Megabytes and save
    peak_mem_mb = peak_mem / (1024 * 1024)

    metrics_list[0]["training_times"].append(trainer.training_time)
    metrics_list[0]["number_of_training_samples"].append(num_samples)
    metrics_list[0]["memory_usage_mb"].append(peak_mem_mb)

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------- LAYER-WISE KERNELS NEIGHBORS GENERATION ----------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for num_samples in TRAINING_SAMPLES:
    
    X_train, Y_train = generate_sinusoidal_tensor(num_samples=num_samples, min_angle=MIN_ANGLE, max_angle=MAX_ANGLE, noise_level=NOISE_LEVEL)

    print(f"\nTraining Data Shape: {X_train.shape}, {Y_train.shape}")

    print(f"\n--- TEST NAME: {TEST_NAME} \t Layer-Wise Kernels Neighbors Generation \t BEAM SEARCH ASTAR Training ---\n")
    model = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE_1),  
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_2, OUTPUT_SIZE),
            )


    trainer = TrainerLayerWiseKernel(model=model,
                            loss_fn=nn.MSELoss(),
                            quantization_factor=QUANTIZATION_FACTOR,
                            parameter_range=PARAMETER_RANGE,
                            debug_mlp=False,
                            #----------------------------------------------------------------------------------
                            weight_kernels = WEIGHT_KERNELS, bias_kernels = BIAS_KERNELS, weight_strides=WEIGHT_STRIDES, bias_strides=BIAS_STRIDES, delta_abs=DELTA_ABS,
                            #----------------------------------------------------------------------------------
                            early_stopping=EARLY_STOPPING, e_s_patience=E_S_PATIENCE,
                            #----------------------------------------------------------------------------------
                            dynamic_quantization=False,
                            #-----------------------------------------------------------------------------------
                            dynamic_kernel_reshaping=False,
                            #----------------------------------------------------------------------------------
                            loss_improvement_threshold=LOSS_IMPROVEMENT_THRESHOLD,
                            #----------------------------------------------------------------------------------
                            max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=SAVE_TRAINED_MODEL, model_name=MODEL_NAME_PREFIX + f'_layer_wise_kernels_astar'
                            )

    # Start tracing memory
    tracemalloc.start()
    
    trainer.beam_search_opt_train(X_train, Y_train, BEAM_WIDTH)
    
    # Capture peak memory and stop tracing
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Convert bytes to Megabytes and save
    peak_mem_mb = peak_mem / (1024 * 1024)

    metrics_list[1]["training_times"].append(trainer.training_time)
    metrics_list[1]["number_of_training_samples"].append(num_samples)
    metrics_list[1]["memory_usage_mb"].append(peak_mem_mb)

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------- RANDOM SAMPLING NEIGHBORS GENERATION -------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for num_samples in TRAINING_SAMPLES:
    
    X_train, Y_train = generate_sinusoidal_tensor(num_samples=num_samples, min_angle=MIN_ANGLE, max_angle=MAX_ANGLE, noise_level=NOISE_LEVEL)

    print(f"\nTraining Data Shape: {X_train.shape}, {Y_train.shape}")

    print(f"\n--- TEST NAME: {TEST_NAME} \t Random Sampling BEAM SEARCH ASTAR Training ---\n")

    model = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE_1),  
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_2, OUTPUT_SIZE),
            )

    trainer = TrainerRandomSampling(model=model,
                            loss_fn=nn.MSELoss(),
                            quantization_factor=QUANTIZATION_FACTOR,
                            parameter_range=PARAMETER_RANGE,
                            debug_mlp=False,
                            #----------------------------------------------------------------------------------
                            perturbation_ratio=PERTURBATION_RATIO, search_coverage_ratio=SEARCH_COVERAGE_RATIO,
                            delta_abs=DELTA_ABS,
                            #----------------------------------------------------------------------------------
                            early_stopping=EARLY_STOPPING, e_s_patience=E_S_PATIENCE,
                            #----------------------------------------------------------------------------------
                            dynamic_quantization=False,
                            #----------------------------------------------------------------------------------
                            loss_improvement_threshold=LOSS_IMPROVEMENT_THRESHOLD,
                            #----------------------------------------------------------------------------------
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
    metrics_list[2]["number_of_training_samples"].append(num_samples)
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
    x_key="number_of_training_samples", x_label="Number of Training Samples",
    y_key="training_times", y_label="Training Time (s)"
)

plot_all_algorithms(
    metrics, 
    output_filename=TEST_NAME + "_time_comparison.png", 
    directory=TEST_NAME,
    x_key="number_of_training_samples", x_label="Number of Training Samples",
    y_key="training_times", y_label="Training Time (s)"
)

# --- MEMORY PLOTS ---
plot_individual_algorithms(
    metrics, 
    prefix=TEST_NAME + "_memory", 
    directory=TEST_NAME, 
    x_key="number_of_training_samples", x_label="Number of Training Samples",
    y_key="memory_usage_mb", y_label="Peak Memory Usage (MB)"
)

plot_all_algorithms(
    metrics, 
    output_filename=TEST_NAME + "_memory_comparison.png", 
    directory=TEST_NAME,
    x_key="number_of_training_samples", x_label="Number of Training Samples",
    y_key="memory_usage_mb", y_label="Peak Memory Usage (MB)"
)