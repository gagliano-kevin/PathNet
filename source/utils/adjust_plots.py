# python -m source.utils.adjust_plots
"""
    This script is used to allow the generation of the plots after the training of the models.
    It uses the functions defined in plot_utils.py to generate the plots and statistical summaries.
    It is ment to be a utility script to adjust the plots after the training without the need to re-train the models.
"""

from source.utils.plot_utils import generate_plots, generate_evaluation_statistical_summary, plot_regression_statistics, load_metrics, json_to_plot_format, format_sci

DATASET_NAME = "California Housing"
TEST_NAME = "small_net_california_housing_SGD_vs_A-star"
NEW_TEST_NAME = "small_net_california_housing_Adam_vs_A-star"


labels_list = ["A-star Single Kernel", "A-star Layer-Wise Kernels", "A-star Random Sampling", "Adam"]

metrics_list = json_to_plot_format(load_metrics(TEST_NAME, directory="results"))

generate_evaluation_statistical_summary(metrics_list,labels_list, NEW_TEST_NAME)

generate_plots(metrics_list, labels_list, NEW_TEST_NAME, DATASET_NAME)

plot_regression_statistics(metrics_list, labels_list, NEW_TEST_NAME, DATASET_NAME)

