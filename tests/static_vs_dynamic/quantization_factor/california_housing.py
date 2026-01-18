#==============================================================================================================================================================
#==============================================================================================================================================================
# python -u -m tests.static_vs_dynamic.quantization_factor.california_housing |& tee tests/static_vs_dynamic/quantization_factor/california_housing_output.txt 
#==============================================================================================================================================================
#==============================================================================================================================================================

#==============================================================================================================================================================
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                   Test on California Housing Dataset to compare Dynamic vs Static Quantization Factor A-star Training
#   In the dynamic case, the quantization factor starts at 10 and is multiplied by 10 every D_Q_PATIENCE iterations without improvement, up to a max of 1e4
#                   In the static cases, all 4 considered values of quantization factors ranging from 1e1 to 1e4 are tested
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#==============================================================================================================================================================


import torch.nn as nn
import math

from source.PathNet import Trainer
from source.utils.dataset_utils.housing_utils import get_california_housing_data, evaluate_pathnet_regression

from source.utils.plot_utils import generate_plots, generate_statistical_summary, format_sci,  save_metrics, load_metrics, generate_regression_statistical_summary, plot_regression_statistics

ITERATIONS = 10
RUNS = 1
SAVE_TRAINED_MODEL = False
MODEL_NAME_PREFIX = "housing_model"
DATASET_NAME = "California Housing"
EARLY_STOPPING = False
E_S_PATIENCE = 200
LOSS_IMPROVEMENT_THRESHOLD = 1e-3

# =========================================================================================================================================================
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------- SMALL NET --------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------
# =========================================================================================================================================================

TEST_NAME = "small_net_quantization_factor"

# MLP Hyperparameters
INPUT_SIZE = 8
HIDDEN_SIZE_1 = 32
HIDDEN_SIZE_2 = 16
OUTPUT_SIZE = 1

PARAMETER_RANGE = (-10, 10)

# Initial Kernel and Stride Settings
WEIGHT_KERNEL = [3,3]
BIAS_KERNEL = [3]
X_STRIDE = 3
Y_STRIDE = 3
DELTA_ABS = None

# Dynamic Quantization Settings
DYNAMIC_QUANTIZATION = True
D_Q_PATIENCE = 100
QUANTIZATION_FACTOR_MULTIPLIER = 10
MIN_QUANTIZATION_FACTOR = 10
MAX_QUANTIZATION_FACTOR = 1e4

DYNAMIC_KERNEL_RESHAPING = False

DYNAMIC_ASTAR_METRICS = {
    "losses": [],
    "training_times": [],
    "final_losses": [],
    "dynamic_quantization_iterations": [],
    "dynamic_kernel_reshaping_iterations": []
}

start_exp = int(math.log10(MIN_QUANTIZATION_FACTOR))
end_exp = int(math.log10(MAX_QUANTIZATION_FACTOR))

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

for qf_exp in range(start_exp, end_exp + 1):
    STATIC_ASTAR_METRICS = {
        "losses": [],
        "training_times": [],
        "final_losses": [],
        "evaluation_scores": []
    }
    metrics_list.append(STATIC_ASTAR_METRICS)

labels_list = ["Dynamic QF [1e1-1e4]"]
for qf_exp in range(start_exp, end_exp + 1):
    labels_list.append(f"Static QF=1e{int(qf_exp)}")
                       
X_train, Y_train, X_val, Y_val, X_test, Y_test = get_california_housing_data()
print(f"\nTraining Data Shape: {X_train.shape}, {Y_train.shape}")
print(f"Validation Data Shape: {X_val.shape}, {Y_val.shape}")
print(f"Testing Data Shape: {X_test.shape}, {Y_test.shape}\n")

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- DYNAMIC ASTAR TRAINING ---------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for run in range(RUNS):
    print(f"\n--- TEST NAME: {TEST_NAME} \t DYNAMIC ASTAR Training Run {run + 1} ---\n")

    model = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE_1),  
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_2, OUTPUT_SIZE),
            )


    trainer = Trainer(model=model,
                             loss_fn=nn.MSELoss(),
                             quantization_factor=MIN_QUANTIZATION_FACTOR,
                             parameter_range=PARAMETER_RANGE,
                             debug_mlp=False,
                             #----------------------------------------------------------------------------------
                             weight_kernel = WEIGHT_KERNEL, bias_kernel = BIAS_KERNEL, x_stride=X_STRIDE, y_stride=Y_STRIDE, delta_abs=DELTA_ABS,
                             #----------------------------------------------------------------------------------
                             early_stopping=EARLY_STOPPING, e_s_patience=E_S_PATIENCE,
                             #----------------------------------------------------------------------------------
                             dynamic_quantization=DYNAMIC_QUANTIZATION, d_q_patience=D_Q_PATIENCE, 
                             quantization_factor_multiplier=QUANTIZATION_FACTOR_MULTIPLIER, max_quantization_factor=MAX_QUANTIZATION_FACTOR,
                             #-----------------------------------------------------------------------------------
                             dynamic_kernel_reshaping=False,
                             #----------------------------------------------------------------------------------
                             loss_improvement_threshold=LOSS_IMPROVEMENT_THRESHOLD,
                             #----------------------------------------------------------------------------------
                             max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=SAVE_TRAINED_MODEL, model_name=MODEL_NAME_PREFIX + f'_dynamic_astar_run_{run + 1}'
                             )

    trainer.train(X_train, Y_train)

    metrics_list[0]["losses"].append(trainer.loss_history)
    metrics_list[0]["training_times"].append(trainer.training_time)
    metrics_list[0]["final_losses"].append(trainer.best_node.h_val)
    metrics_list[0]["dynamic_quantization_iterations"].append(trainer.dynamic_adjustments_log["dynamic_quantization_iterations"])
    metrics_list[0]["dynamic_kernel_reshaping_iterations"].append(trainer.dynamic_adjustments_log["dynamic_kernel_reshaping_iterations"])
    metrics_list[0]["evaluation_scores"].append(evaluate_pathnet_regression(trainer, (X_test, Y_test)))


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- STATIC ASTAR TRAININGS ---------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for index, exponent in enumerate(range(start_exp, end_exp + 1)):
    quantization_factor = 10 ** exponent
    for run in range(RUNS):
        print(f"\n--- TEST NAME: {TEST_NAME} \t STATIC ASTAR Training Run {run + 1} \t QUANTIZATION FACTOR: {format_sci(quantization_factor)} ---\n")

        model = nn.Sequential(
                nn.Linear(INPUT_SIZE, HIDDEN_SIZE_1),  
                nn.ReLU(),
                nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2),
                nn.ReLU(),
                nn.Linear(HIDDEN_SIZE_2, OUTPUT_SIZE),
                )


        trainer = Trainer(model=model,
                                 loss_fn=nn.MSELoss(),
                                 quantization_factor=quantization_factor,
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
                                 max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=SAVE_TRAINED_MODEL, model_name=MODEL_NAME_PREFIX + f'_static_astar_qf_{format_sci(quantization_factor)}_run_{run + 1}'
                                 )

        trainer.train(X_train, Y_train)

        metrics_list[index + 1]["losses"].append(trainer.loss_history)
        metrics_list[index + 1]["training_times"].append(trainer.training_time)
        metrics_list[index + 1]["final_losses"].append(trainer.best_node.h_val)
        metrics_list[index + 1]["evaluation_scores"].append(evaluate_pathnet_regression(trainer, (X_test, Y_test)))


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- COMPARISON ---------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------


generate_regression_statistical_summary(metrics_list,labels_list, TEST_NAME)

generate_plots(metrics_list, labels_list, TEST_NAME, DATASET_NAME)

plot_regression_statistics(metrics_list, labels_list, TEST_NAME, DATASET_NAME)

all_results = {label: metric for label, metric in zip(labels_list, metrics_list)}

save_metrics(all_results, TEST_NAME)




# =========================================================================================================================================================
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------- MEDIUM NET --------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------
# =========================================================================================================================================================

TEST_NAME = "medium_net_quantization_factor"

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

for qf_exp in range(start_exp, end_exp + 1):
    STATIC_ASTAR_METRICS = {
        "losses": [],
        "training_times": [],
        "final_losses": [],
        "evaluation_scores": []
    }
    metrics_list.append(STATIC_ASTAR_METRICS)

labels_list = ["Dynamic QF [1e1-1e4]"]
for qf_exp in range(start_exp, end_exp + 1):
    labels_list.append(f"Static QF=1e{int(qf_exp)}")


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- DYNAMIC ASTAR TRAINING ---------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for run in range(RUNS):
    print(f"\n--- TEST NAME: {TEST_NAME} \t DYNAMIC ASTAR Training Run {run + 1} ---\n")

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
                             quantization_factor=MIN_QUANTIZATION_FACTOR,
                             parameter_range=PARAMETER_RANGE,
                             debug_mlp=False,
                             #----------------------------------------------------------------------------------
                             weight_kernel = WEIGHT_KERNEL, bias_kernel = BIAS_KERNEL, x_stride=X_STRIDE, y_stride=Y_STRIDE, delta_abs=DELTA_ABS,
                             #----------------------------------------------------------------------------------
                             early_stopping=EARLY_STOPPING, e_s_patience=E_S_PATIENCE,
                             #----------------------------------------------------------------------------------
                             dynamic_quantization=DYNAMIC_QUANTIZATION, d_q_patience=D_Q_PATIENCE, 
                             quantization_factor_multiplier=QUANTIZATION_FACTOR_MULTIPLIER, max_quantization_factor=MAX_QUANTIZATION_FACTOR,
                             #-----------------------------------------------------------------------------------
                             dynamic_kernel_reshaping=False,
                             #----------------------------------------------------------------------------------
                             loss_improvement_threshold=LOSS_IMPROVEMENT_THRESHOLD,
                             #----------------------------------------------------------------------------------
                             max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=SAVE_TRAINED_MODEL, model_name=MODEL_NAME_PREFIX + f'_dynamic_astar_run_{run + 1}'
                             )

    trainer.train(X_train, Y_train)

    metrics_list[0]["losses"].append(trainer.loss_history)
    metrics_list[0]["training_times"].append(trainer.training_time)
    metrics_list[0]["final_losses"].append(trainer.best_node.h_val)
    metrics_list[0]["dynamic_quantization_iterations"].append(trainer.dynamic_adjustments_log["dynamic_quantization_iterations"])
    metrics_list[0]["dynamic_kernel_reshaping_iterations"].append(trainer.dynamic_adjustments_log["dynamic_kernel_reshaping_iterations"])
    metrics_list[0]["evaluation_scores"].append(evaluate_pathnet_regression(trainer, (X_test, Y_test)))


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- STATIC ASTAR TRAININGS ---------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for index, exponent in enumerate(range(start_exp, end_exp + 1)):
    quantization_factor = 10 ** exponent
    for run in range(RUNS):
        print(f"\n--- TEST NAME: {TEST_NAME} \t STATIC ASTAR Training Run {run + 1} \t QUANTIZATION FACTOR: {format_sci(quantization_factor)} ---\n")

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
                                 quantization_factor=quantization_factor,
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
                                 max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=SAVE_TRAINED_MODEL, model_name=MODEL_NAME_PREFIX + f'_static_astar_qf_{format_sci(quantization_factor)}_run_{run + 1}'
                                 )

        trainer.train(X_train, Y_train)

        metrics_list[index + 1]["losses"].append(trainer.loss_history)
        metrics_list[index + 1]["training_times"].append(trainer.training_time)
        metrics_list[index + 1]["final_losses"].append(trainer.best_node.h_val)
        metrics_list[index + 1]["evaluation_scores"].append(evaluate_pathnet_regression(trainer, (X_test, Y_test)))


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- COMPARISON ---------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------


generate_regression_statistical_summary(metrics_list,labels_list, TEST_NAME)

generate_plots(metrics_list, labels_list, TEST_NAME, DATASET_NAME)

plot_regression_statistics(metrics_list, labels_list, TEST_NAME, DATASET_NAME)

all_results = {label: metric for label, metric in zip(labels_list, metrics_list)}

save_metrics(all_results, TEST_NAME)



# =========================================================================================================================================================
#----------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------- BIG NET ----------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------
# =========================================================================================================================================================

TEST_NAME = "big_net_quantization_factor"

# MLP Hyperparameters
HIDDEN_SIZE_1 = 128
HIDDEN_SIZE_2 = 64
HIDDEN_SIZE_3 = 32
HIDDEN_SIZE_4 = 16

# Initial Kernel and Stride Settings
WEIGHT_KERNEL = [5,5]
BIAS_KERNEL = [5]
X_STRIDE = 5
Y_STRIDE = 5

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

for qf_exp in range(start_exp, end_exp + 1):
    STATIC_ASTAR_METRICS = {
        "losses": [],
        "training_times": [],
        "final_losses": [],
        "evaluation_scores": []
    }
    metrics_list.append(STATIC_ASTAR_METRICS)

labels_list = ["Dynamic QF [1e1-1e4]"]
for qf_exp in range(start_exp, end_exp + 1):
    labels_list.append(f"Static QF=1e{int(qf_exp)}")


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- DYNAMIC ASTAR TRAINING ---------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for run in range(RUNS):
    print(f"\n--- TEST NAME: {TEST_NAME} \t DYNAMIC ASTAR Training Run {run + 1} ---\n")

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
                             quantization_factor=MIN_QUANTIZATION_FACTOR,
                             parameter_range=PARAMETER_RANGE,
                             debug_mlp=False,
                             #----------------------------------------------------------------------------------
                             weight_kernel = WEIGHT_KERNEL, bias_kernel = BIAS_KERNEL, x_stride=X_STRIDE, y_stride=Y_STRIDE, delta_abs=DELTA_ABS,
                             #----------------------------------------------------------------------------------
                             early_stopping=EARLY_STOPPING, e_s_patience=E_S_PATIENCE,
                             #----------------------------------------------------------------------------------
                             dynamic_quantization=DYNAMIC_QUANTIZATION, d_q_patience=D_Q_PATIENCE, 
                             quantization_factor_multiplier=QUANTIZATION_FACTOR_MULTIPLIER, max_quantization_factor=MAX_QUANTIZATION_FACTOR,
                             #-----------------------------------------------------------------------------------
                             dynamic_kernel_reshaping=False,
                             #----------------------------------------------------------------------------------
                             loss_improvement_threshold=LOSS_IMPROVEMENT_THRESHOLD,
                             #----------------------------------------------------------------------------------
                             max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=SAVE_TRAINED_MODEL, model_name=MODEL_NAME_PREFIX + f'_dynamic_astar_run_{run + 1}'
                             )

    trainer.train(X_train, Y_train)

    metrics_list[0]["losses"].append(trainer.loss_history)
    metrics_list[0]["training_times"].append(trainer.training_time)
    metrics_list[0]["final_losses"].append(trainer.best_node.h_val)
    metrics_list[0]["dynamic_quantization_iterations"].append(trainer.dynamic_adjustments_log["dynamic_quantization_iterations"])
    metrics_list[0]["dynamic_kernel_reshaping_iterations"].append(trainer.dynamic_adjustments_log["dynamic_kernel_reshaping_iterations"])
    metrics_list[0]["evaluation_scores"].append(evaluate_pathnet_regression(trainer, (X_test, Y_test)))


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- STATIC ASTAR TRAININGS ---------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------

for index, exponent in enumerate(range(start_exp, end_exp + 1)):
    quantization_factor = 10 ** exponent
    for run in range(RUNS):
        print(f"\n--- TEST NAME: {TEST_NAME} \t STATIC ASTAR Training Run {run + 1} \t QUANTIZATION FACTOR: {format_sci(quantization_factor)} ---\n")

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
                                 quantization_factor=quantization_factor,
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
                                 max_iterations=ITERATIONS, log_freq=100, measure_time=True, save_trained_model=SAVE_TRAINED_MODEL, model_name=MODEL_NAME_PREFIX + f'_static_astar_qf_{format_sci(quantization_factor)}_run_{run + 1}'
                                 )

        trainer.train(X_train, Y_train)

        metrics_list[index + 1]["losses"].append(trainer.loss_history)
        metrics_list[index + 1]["training_times"].append(trainer.training_time)
        metrics_list[index + 1]["final_losses"].append(trainer.best_node.h_val)
        metrics_list[index + 1]["evaluation_scores"].append(evaluate_pathnet_regression(trainer, (X_test, Y_test)))


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------- COMPARISON ---------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------


generate_regression_statistical_summary(metrics_list,labels_list, TEST_NAME)

generate_plots(metrics_list, labels_list, TEST_NAME, DATASET_NAME)

plot_regression_statistics(metrics_list, labels_list, TEST_NAME, DATASET_NAME)

all_results = {label: metric for label, metric in zip(labels_list, metrics_list)}

save_metrics(all_results, TEST_NAME)
