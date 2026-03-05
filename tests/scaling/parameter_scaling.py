#==============================================================================================================================================================
#==============================================================================================================================================================
#-------------------------------------------------- python -m tests.scaling.parameter_scaling ------------------------------------------------------
#==============================================================================================================================================================
#==============================================================================================================================================================

import torch
import torch.nn as nn
import numpy as np

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
NUM_SAMPLES = 1000
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

TEST_NAME = "sine_parameter_scaling"

# Architectures to test: List of tuples (HIDDEN_SIZE_1, HIDDEN_SIZE_2)
# Scaling up the hidden dimensions will smoothly scale the total parameter count
NETWORK_ARCHITECTURES = [
    (8, 4),    
    (16, 8),   
    (32, 16),  
    (64, 32),
    (128, 64),
    (256, 128)
]

# Parameter for single Kernel Neighbors Generation
WEIGHT_KERNEL = [2,2]
BIAS_KERNEL = [2]
X_STRIDE = 1
Y_STRIDE = 1

# Parameters for Layer-Wise Kernels Neighbors Generation
# Note: This assumes 3 layers (Linear -> Linear -> Linear). 
# If the number of layers changes, these lists must be adjusted.
WEIGHT_KERNELS = [[2,2], [2,2], [1,2]]
BIAS_KERNELS = [[2], [2], [1]]
WEIGHT_STRIDES = [[1,1], [1,1], [1,1]]      
BIAS_STRIDES = [[1], [1], [1]]              

# Parameters for Random Sampling Neighbors Generation
PERTURBATION_RATIO = 0.01       
SEARCH_COVERAGE_RATIO = 0.1     

metrics_list = [
    {
        "training_times": [],
        "number_of_parameters": [],
    } for _ in range(len(labels_list))
]

# Helper function to count parameters
count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------- SINGLE KERNEL NEIGHBORS GENERATION ------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

# Generate dataset once for all models to ensure identical training conditions
X_train, Y_train = generate_sinusoidal_tensor(num_samples=NUM_SAMPLES, min_angle=MIN_ANGLE, max_angle=MAX_ANGLE, noise_level=NOISE_LEVEL)
print(f"\nTraining Data Shape: {X_train.shape}, {Y_train.shape}")

for hidden_1, hidden_2 in NETWORK_ARCHITECTURES:

    print(f"\n--- TEST NAME: {TEST_NAME} \t Single Kernel \t Hidden Sizes: ({hidden_1}, {hidden_2}) ---\n")

    model = nn.Sequential(
            nn.Linear(INPUT_SIZE, hidden_1),  
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, OUTPUT_SIZE),
            )
            
    num_params = count_parameters(model)

    trainer = Trainer(model=model,
                            loss_fn=nn.MSELoss(),
                            quantization_factor=QUANTIZATION_FACTOR,
                            parameter_range=PARAMETER_RANGE,
                            debug_mlp=False,
                            weight_kernel = WEIGHT_KERNEL, bias_kernel = BIAS_KERNEL, x_stride=X_STRIDE, y_stride=Y_STRIDE, delta_abs=DELTA_ABS,
                            early_stopping=EARLY_STOPPING, e_s_patience=E_S_PATIENCE,
                            dynamic_quantization=False,
                            dynamic_kernel_reshaping=False,
                            loss_improvement_threshold=LOSS_IMPROVEMENT_THRESHOLD,
                            max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=SAVE_TRAINED_MODEL, model_name=MODEL_NAME_PREFIX + f'_single_kernel_astar'
                            )

    trainer.beam_search_opt_train(X_train, Y_train, BEAM_WIDTH)

    metrics_list[0]["training_times"].append(trainer.training_time)
    metrics_list[0]["number_of_parameters"].append(num_params)

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------- LAYER-WISE KERNELS NEIGHBORS GENERATION ----------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for hidden_1, hidden_2 in NETWORK_ARCHITECTURES:

    print(f"\n--- TEST NAME: {TEST_NAME} \t Layer-Wise Kernels \t Hidden Sizes: ({hidden_1}, {hidden_2}) ---\n")
    
    model = nn.Sequential(
            nn.Linear(INPUT_SIZE, hidden_1),  
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, OUTPUT_SIZE),
            )

    num_params = count_parameters(model)

    trainer = TrainerLayerWiseKernel(model=model,
                            loss_fn=nn.MSELoss(),
                            quantization_factor=QUANTIZATION_FACTOR,
                            parameter_range=PARAMETER_RANGE,
                            debug_mlp=False,
                            weight_kernels = WEIGHT_KERNELS, bias_kernels = BIAS_KERNELS, weight_strides=WEIGHT_STRIDES, bias_strides=BIAS_STRIDES, delta_abs=DELTA_ABS,
                            early_stopping=EARLY_STOPPING, e_s_patience=E_S_PATIENCE,
                            dynamic_quantization=False,
                            dynamic_kernel_reshaping=False,
                            loss_improvement_threshold=LOSS_IMPROVEMENT_THRESHOLD,
                            max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=SAVE_TRAINED_MODEL, model_name=MODEL_NAME_PREFIX + f'_layer_wise_kernels_astar'
                            )

    trainer.beam_search_opt_train(X_train, Y_train, BEAM_WIDTH)

    metrics_list[1]["training_times"].append(trainer.training_time)
    metrics_list[1]["number_of_parameters"].append(num_params)

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

    trainer.beam_search_opt_train(X_train, Y_train, BEAM_WIDTH)

    metrics_list[2]["training_times"].append(trainer.training_time)
    metrics_list[2]["number_of_parameters"].append(num_params)

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- COMPARISON ---------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

all_results = {label: metric for label, metric in zip(labels_list, metrics_list)}

save_metrics(all_results, TEST_NAME)

metrics = load_metrics(TEST_NAME)

plot_individual_algorithms(
    metrics, 
    prefix=TEST_NAME, 
    directory=TEST_NAME, 
    x_key="number_of_parameters", 
    x_label="Number of Model Parameters"
)

plot_all_algorithms(
    metrics, 
    output_filename=TEST_NAME + "_comparison.png", 
    directory=TEST_NAME,
    x_key="number_of_parameters", 
    x_label="Number of Model Parameters"
)
