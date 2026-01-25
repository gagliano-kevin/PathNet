#==============================================================================================================================================================
#==============================================================================================================================================================
# python -u -m tests.vanilla_train_vs_beam_search.california_housing |& tee tests/vanilla_train_vs_beam_search/california_housing_output.txt 
#==============================================================================================================================================================
#==============================================================================================================================================================

#==============================================================================================================================================================
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                   Test on California Housing Dataset to compare standard astr training versus a possible optimized beam search training.
#                               The test is performed on three different MLP architectures: small, medium, and big.
#                                           For each architecture, two training approaches are compared:
#                           1. Standard A* Training with fixed quantization factors (static) and fixed kernel/stride settings.
#                           2. Beam Search Training with fixed quantization factors (static) and fixed kernel/stride settings.
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#==============================================================================================================================================================


import torch.nn as nn

from source.PathNet import Trainer
from source.utils.dataset_utils.housing_utils import get_california_housing_data
from source.utils.evaluation_utils import evaluate_pathnet_regression
from source.utils.plot_utils import generate_plots, format_sci,  save_metrics, generate_evaluation_statistical_summary, plot_regression_statistics

ITERATIONS = 2000
RUNS = 5
SAVE_TRAINED_MODEL = False
MODEL_NAME_PREFIX = "housing_model"
DATASET_NAME = "California Housing"
EARLY_STOPPING = False
E_S_PATIENCE = 200
LOSS_IMPROVEMENT_THRESHOLD = 1e-3
PARAMETER_RANGE = (-10, 10)

BEAM_WIDTHS = [1e2, 1e3, 1e4]


# =========================================================================================================================================================
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------- SMALL NET --------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------
# =========================================================================================================================================================

TEST_NAME = "small_net_vanilla_train_vs_beam_search"

# MLP Hyperparameters
INPUT_SIZE = 8
HIDDEN_SIZE_1 = 32
HIDDEN_SIZE_2 = 16
OUTPUT_SIZE = 1

# Initial Kernel and Stride Settings
WEIGHT_KERNEL = [2,2]
BIAS_KERNEL = [2]
X_STRIDE = 2
Y_STRIDE = 2
DELTA_ABS = None

# Quantization Parameter
QUANTIZATION_FACTOR = 10

metrics_list = [
    {
        "losses": [],
        "training_times": [],
        "final_losses": [],
        "dynamic_quantization_iterations": [],
        "dynamic_kernel_reshaping_iterations": [],
        "evaluation_scores": []
    }
]

for beam_width in BEAM_WIDTHS:
    BW_ASTAR_METRICS = {
        "losses": [],
        "training_times": [],
        "final_losses": [],
        "dynamic_quantization_iterations": [],
        "dynamic_kernel_reshaping_iterations": [],
        "evaluation_scores": []
    }
    metrics_list.append(BW_ASTAR_METRICS)

labels_list = ["Standard A*"]
for beam_width in BEAM_WIDTHS:
    labels_list.append(f"Beam Search BW={format_sci(beam_width)}")
                       
X_train, Y_train, X_val, Y_val, X_test, Y_test = get_california_housing_data()
print(f"\nTraining Data Shape: {X_train.shape}, {Y_train.shape}")
print(f"Validation Data Shape: {X_val.shape}, {Y_val.shape}")
print(f"Testing Data Shape: {X_test.shape}, {Y_test.shape}\n")

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- VANILLA ASTAR TRAINING ---------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for run in range(RUNS):
    print(f"\n--- TEST NAME: {TEST_NAME} \t Vanilla ASTAR Training Run {run + 1} ---\n")

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
                            max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=SAVE_TRAINED_MODEL, model_name=MODEL_NAME_PREFIX + f'_vanilla_astar_run_{run + 1}'
                            )

    trainer.train(X_train, Y_train)

    metrics_list[0]["losses"].append(trainer.loss_history)
    metrics_list[0]["training_times"].append(trainer.training_time)
    metrics_list[0]["final_losses"].append(trainer.best_node.h_val)
    metrics_list[0]["dynamic_quantization_iterations"].append(trainer.dynamic_adjustments_log["dynamic_quantization_iterations"])
    metrics_list[0]["dynamic_kernel_reshaping_iterations"].append(trainer.dynamic_adjustments_log["dynamic_kernel_reshaping_iterations"])
    metrics_list[0]["evaluation_scores"].append(evaluate_pathnet_regression(trainer, (X_test, Y_test)))


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------- BEAM SEARCH ASTAR TRAININGS ---------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for index, beam_width in enumerate(BEAM_WIDTHS):

    for run in range(RUNS):
        print(f"\n--- TEST NAME: {TEST_NAME} \t BEAM SEARCH Training Run {run + 1} \t BEAM WIDTH: {format_sci(beam_width)} ---\n")

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
                                max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=SAVE_TRAINED_MODEL, model_name=MODEL_NAME_PREFIX + f'_beam_search_astar_run_{run + 1}'
                                )

        trainer.beam_search_opt_train(X_train, Y_train, beam_width)

        metrics_list[index + 1]["losses"].append(trainer.loss_history)
        metrics_list[index + 1]["training_times"].append(trainer.training_time)
        metrics_list[index + 1]["final_losses"].append(trainer.best_node.h_val)
        metrics_list[index + 1]["dynamic_quantization_iterations"].append(trainer.dynamic_adjustments_log["dynamic_quantization_iterations"])
        metrics_list[index + 1]["dynamic_kernel_reshaping_iterations"].append(trainer.dynamic_adjustments_log["dynamic_kernel_reshaping_iterations"])
        metrics_list[index + 1]["evaluation_scores"].append(evaluate_pathnet_regression(trainer, (X_test, Y_test)))


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- COMPARISON ---------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

all_results = {label: metric for label, metric in zip(labels_list, metrics_list)}

save_metrics(all_results, TEST_NAME)

generate_evaluation_statistical_summary(metrics_list,labels_list, TEST_NAME)

generate_plots(metrics_list, labels_list, TEST_NAME, DATASET_NAME)

plot_regression_statistics(metrics_list, labels_list, TEST_NAME, DATASET_NAME)



# =========================================================================================================================================================
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------- MEDIUM NET --------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------
# =========================================================================================================================================================

TEST_NAME = "medium_net_vanilla_train_vs_beam_search"

# MLP Hyperparameters
HIDDEN_SIZE_1 = 64
HIDDEN_SIZE_2 = 32
HIDDEN_SIZE_3 = 16

# Initial Kernel and Stride Settings
WEIGHT_KERNEL = [4,4]
BIAS_KERNEL = [4]
X_STRIDE = 4
Y_STRIDE = 4


metrics_list = [
    {
        "losses": [],
        "training_times": [],
        "final_losses": [],
        "dynamic_quantization_iterations": [],
        "dynamic_kernel_reshaping_iterations": [],
        "evaluation_scores": []
    }
]

for beam_width in BEAM_WIDTHS:
    BW_ASTAR_METRICS = {
        "losses": [],
        "training_times": [],
        "final_losses": [],
        "dynamic_quantization_iterations": [],
        "dynamic_kernel_reshaping_iterations": [],
        "evaluation_scores": []
    }
    metrics_list.append(BW_ASTAR_METRICS)

labels_list = ["Standard A*"]
for beam_width in BEAM_WIDTHS:
    labels_list.append(f"Beam Search BW={format_sci(beam_width)}")


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- VANILLA ASTAR TRAINING ---------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for run in range(RUNS):
    print(f"\n--- TEST NAME: {TEST_NAME} \t Vanilla ASTAR Training Run {run + 1} ---\n")

    model = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE_1),  
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_2, HIDDEN_SIZE_3),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_3, OUTPUT_SIZE),
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
                            max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=SAVE_TRAINED_MODEL, model_name=MODEL_NAME_PREFIX + f'_vanilla_astar_run_{run + 1}'
                            )

    trainer.train(X_train, Y_train)

    metrics_list[0]["losses"].append(trainer.loss_history)
    metrics_list[0]["training_times"].append(trainer.training_time)
    metrics_list[0]["final_losses"].append(trainer.best_node.h_val)
    metrics_list[0]["dynamic_quantization_iterations"].append(trainer.dynamic_adjustments_log["dynamic_quantization_iterations"])
    metrics_list[0]["dynamic_kernel_reshaping_iterations"].append(trainer.dynamic_adjustments_log["dynamic_kernel_reshaping_iterations"])
    metrics_list[0]["evaluation_scores"].append(evaluate_pathnet_regression(trainer, (X_test, Y_test)))


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------- BEAM SEARCH ASTAR TRAININGS ---------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for index, beam_width in enumerate(BEAM_WIDTHS):

    for run in range(RUNS):
        print(f"\n--- TEST NAME: {TEST_NAME} \t BEAM SEARCH Training Run {run + 1} \t BEAM WIDTH: {format_sci(beam_width)} ---\n")

        model = nn.Sequential(
                nn.Linear(INPUT_SIZE, HIDDEN_SIZE_1),  
                nn.ReLU(),
                nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2),
                nn.ReLU(),
                nn.Linear(HIDDEN_SIZE_2, HIDDEN_SIZE_3),
                nn.ReLU(),
                nn.Linear(HIDDEN_SIZE_3, OUTPUT_SIZE),
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
                                max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=SAVE_TRAINED_MODEL, model_name=MODEL_NAME_PREFIX + f'_beam_search_astar_run_{run + 1}'
                                )

        trainer.beam_search_opt_train(X_train, Y_train, beam_width)

        metrics_list[index + 1]["losses"].append(trainer.loss_history)
        metrics_list[index + 1]["training_times"].append(trainer.training_time)
        metrics_list[index + 1]["final_losses"].append(trainer.best_node.h_val)
        metrics_list[index + 1]["dynamic_quantization_iterations"].append(trainer.dynamic_adjustments_log["dynamic_quantization_iterations"])
        metrics_list[index + 1]["dynamic_kernel_reshaping_iterations"].append(trainer.dynamic_adjustments_log["dynamic_kernel_reshaping_iterations"])
        metrics_list[index + 1]["evaluation_scores"].append(evaluate_pathnet_regression(trainer, (X_test, Y_test)))


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- COMPARISON ---------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

all_results = {label: metric for label, metric in zip(labels_list, metrics_list)}

save_metrics(all_results, TEST_NAME)

generate_evaluation_statistical_summary(metrics_list,labels_list, TEST_NAME)

generate_plots(metrics_list, labels_list, TEST_NAME, DATASET_NAME)

plot_regression_statistics(metrics_list, labels_list, TEST_NAME, DATASET_NAME)



# =========================================================================================================================================================
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------- BIG NET ----------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------
# =========================================================================================================================================================

TEST_NAME = "big_net_vanilla_train_vs_beam_search"

# MLP Hyperparameters
HIDDEN_SIZE_1 = 128
HIDDEN_SIZE_2 = 64
HIDDEN_SIZE_3 = 32
HIDDEN_SIZE_4 = 16

# Initial Kernel and Stride Settings
WEIGHT_KERNEL = [8,8]
BIAS_KERNEL = [8]
X_STRIDE = 8
Y_STRIDE = 8

metrics_list = [
    {
        "losses": [],
        "training_times": [],
        "final_losses": [],
        "dynamic_quantization_iterations": [],
        "dynamic_kernel_reshaping_iterations": [],
        "evaluation_scores": []
    }
]

for beam_width in BEAM_WIDTHS:
    BW_ASTAR_METRICS = {
        "losses": [],
        "training_times": [],
        "final_losses": [],
        "dynamic_quantization_iterations": [],
        "dynamic_kernel_reshaping_iterations": [],
        "evaluation_scores": []
    }
    metrics_list.append(BW_ASTAR_METRICS)

labels_list = ["Standard A*"]
for beam_width in BEAM_WIDTHS:
    labels_list.append(f"Beam Search BW={format_sci(beam_width)}")


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- VANILLA ASTAR TRAINING ---------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for run in range(RUNS):
    print(f"\n--- TEST NAME: {TEST_NAME} \t Vanilla ASTAR Training Run {run + 1} ---\n")

    model = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE_1),  
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_2, HIDDEN_SIZE_3),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_3, HIDDEN_SIZE_4),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_4, OUTPUT_SIZE),
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
                            max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=SAVE_TRAINED_MODEL, model_name=MODEL_NAME_PREFIX + f'_vanilla_astar_run_{run + 1}'
                            )

    trainer.train(X_train, Y_train)

    metrics_list[0]["losses"].append(trainer.loss_history)
    metrics_list[0]["training_times"].append(trainer.training_time)
    metrics_list[0]["final_losses"].append(trainer.best_node.h_val)
    metrics_list[0]["dynamic_quantization_iterations"].append(trainer.dynamic_adjustments_log["dynamic_quantization_iterations"])
    metrics_list[0]["dynamic_kernel_reshaping_iterations"].append(trainer.dynamic_adjustments_log["dynamic_kernel_reshaping_iterations"])
    metrics_list[0]["evaluation_scores"].append(evaluate_pathnet_regression(trainer, (X_test, Y_test)))


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------- BEAM SEARCH ASTAR TRAININGS ---------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for index, beam_width in enumerate(BEAM_WIDTHS):

    for run in range(RUNS):
        print(f"\n--- TEST NAME: {TEST_NAME} \t BEAM SEARCH Training Run {run + 1} \t BEAM WIDTH: {format_sci(beam_width)} ---\n")

        model = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE_1),  
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_2, HIDDEN_SIZE_3),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_3, HIDDEN_SIZE_4),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_4, OUTPUT_SIZE),
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
                                max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=SAVE_TRAINED_MODEL, model_name=MODEL_NAME_PREFIX + f'_beaam_search_astar_run_{run + 1}'
                                )

        trainer.beam_search_opt_train(X_train, Y_train, beam_width)

        metrics_list[index + 1]["losses"].append(trainer.loss_history)
        metrics_list[index + 1]["training_times"].append(trainer.training_time)
        metrics_list[index + 1]["final_losses"].append(trainer.best_node.h_val)
        metrics_list[index + 1]["dynamic_quantization_iterations"].append(trainer.dynamic_adjustments_log["dynamic_quantization_iterations"])
        metrics_list[index + 1]["dynamic_kernel_reshaping_iterations"].append(trainer.dynamic_adjustments_log["dynamic_kernel_reshaping_iterations"])
        metrics_list[index + 1]["evaluation_scores"].append(evaluate_pathnet_regression(trainer, (X_test, Y_test)))


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- COMPARISON ---------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

all_results = {label: metric for label, metric in zip(labels_list, metrics_list)}

save_metrics(all_results, TEST_NAME)

generate_evaluation_statistical_summary(metrics_list,labels_list, TEST_NAME)

generate_plots(metrics_list, labels_list, TEST_NAME, DATASET_NAME)

plot_regression_statistics(metrics_list, labels_list, TEST_NAME, DATASET_NAME)
