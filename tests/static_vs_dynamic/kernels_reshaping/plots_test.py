# python -m tests.static_vs_dynamic.kernels_reshaping.plots_test
"""
    This test script verifies the plotting utilities for regression tasks, using the json metrics
    generated from the dynamic vs static kernels reshaping experiments on the California Housing dataset.
    It generates evaluation statistical summaries and visual plots to compare model performances.
"""

from source.utils.plot_utils import generate_plots, generate_evaluation_statistical_summary, plot_regression_statistics, load_metrics, json_to_plot_format


DATASET_NAME = "California Housing"
TEST_NAME = "small_net_dynamic_kernels_reshaping"

# Initial Kernel and Stride Settings
MAX_WEIGHT_KERNEL = [4,4]
MAX_BIAS_KERNEL = [4]
MIN_WEIGHT_KERNEL = [2,2]


labels_list = [f"Dyn. KS=[{MAX_WEIGHT_KERNEL[0]}x{MAX_WEIGHT_KERNEL[1]}-{MIN_WEIGHT_KERNEL[0]}x{MIN_WEIGHT_KERNEL[1]}]"]
for kernel_size in range(MAX_WEIGHT_KERNEL[0], MIN_WEIGHT_KERNEL[0] - 1, -1):
    labels_list.append(f"Stat. KS={kernel_size}x{kernel_size}")
                       
metrics_list = json_to_plot_format(load_metrics(TEST_NAME))

generate_evaluation_statistical_summary(metrics_list,labels_list, TEST_NAME)

generate_plots(metrics_list, labels_list, TEST_NAME, DATASET_NAME)

plot_regression_statistics(metrics_list, labels_list, TEST_NAME, DATASET_NAME)

