import numpy as np
from source.general_utils import plot_final_loss_distribution, plot_mean_loss_with_std, pad_to_max
import warnings


def generate_statistical_summary(static_dict, dynamic_dict, filename):
    # FINAL LOSS STATS (for Summary Table and Box Plot)
    RUNS = len(static_dict["final_losses"])
    static_astar_final_losses = np.array(static_dict["final_losses"])
    static_astar_training_times = np.array(static_dict["training_times"])
    dynamic_astar_final_losses = np.array(dynamic_dict["final_losses"])
    dynamic_astar_training_times = np.array(dynamic_dict["training_times"])


    # STATIC A-Star Statistics
    static_astar_avg_loss = np.mean(static_astar_final_losses)
    static_astar_std_dev = np.std(static_astar_final_losses)
    static_astar_variance = np.var(static_astar_final_losses)
    static_astar_median = np.median(static_astar_final_losses)
    static_astar_min = np.min(static_astar_final_losses)
    static_astar_max = np.max(static_astar_final_losses)
    static_astar_avg_training_time = np.mean(static_astar_training_times)

    # DYNAMIC A-Star Statistics
    dynamic_astar_avg_loss = np.mean(dynamic_astar_final_losses)
    dynamic_astar_std_dev = np.std(dynamic_astar_final_losses)
    dynamic_astar_variance = np.var(dynamic_astar_final_losses)
    dynamic_astar_median = np.median(dynamic_astar_final_losses)
    dynamic_astar_min = np.min(dynamic_astar_final_losses)
    dynamic_astar_max = np.max(dynamic_astar_final_losses)
    dynamic_astar_avg_training_time = np.mean(dynamic_astar_training_times)


    print("\n=========================================================================================")
    print(f"| STATISTICAL SUMMARY over {RUNS} Runs |")
    print("=========================================================================================")
    print("| Metric      | STATIC A-Star  | DYNAMIC A-Star |")
    print("|-------------|----------------|----------------------------|")
    print(f"| Average Loss| {static_astar_avg_loss:.6f}     | {dynamic_astar_avg_loss:.6f}              |")
    print(f"| Median Loss | {static_astar_median:.6f}     | {dynamic_astar_median:.6f}              |")
    print(f"| Std Dev     | {static_astar_std_dev:.6f}     | {dynamic_astar_std_dev:.6f}              |")
    print(f"| Variance    | {static_astar_variance:.6f}     | {dynamic_astar_variance:.6f}              |")
    print(f"| Min Loss    | {static_astar_min:.6f}     | {dynamic_astar_min:.6f}              |")
    print(f"| Max Loss    | {static_astar_max:.6f}     | {dynamic_astar_max:.6f}              |")
    print(f"| AVG Training Time | {static_astar_avg_training_time:.6f} | {dynamic_astar_avg_training_time:.6f}         |")
    print("=========================================================================================")

    with open(f"{filename}.txt", "w") as f:
        f.write("=========================================================================================\n")
        f.write(f"| STATISTICAL SUMMARY over {RUNS} Runs |\n")
        f.write("=========================================================================================\n")
        f.write("| Metric      | STATIC A-Star  | DYNAMIC A-Star |\n")
        f.write("|-------------|----------------|----------------------------|\n")
        f.write(f"| Average Loss| {static_astar_avg_loss:.6f}     | {dynamic_astar_avg_loss:.6f}              |\n")
        f.write(f"| Median Loss | {static_astar_median:.6f}     | {dynamic_astar_median:.6f}              |\n")
        f.write(f"| Std Dev     | {static_astar_std_dev:.6f}     | {dynamic_astar_std_dev:.6f}              |\n")
        f.write(f"| Variance    | {static_astar_variance:.6f}     | {dynamic_astar_variance:.6f}              |\n")
        f.write(f"| Min Loss    | {static_astar_min:.6f}     | {dynamic_astar_min:.6f}              |\n")
        f.write(f"| Max Loss    | {static_astar_max:.6f}     | {dynamic_astar_max:.6f}              |\n")
        f.write(f"| AVG Training Time | {static_astar_avg_training_time:.6f} | {dynamic_astar_avg_training_time:.6f}         |\n")
        f.write("=========================================================================================\n")

    print(f"\nSaved statistical summary to '{filename}.txt'\n")



def generate_plots(static_dict, dynamic_dict, filename):

    RUNS = len(static_dict["final_losses"])

    static_astar_losses_array = static_dict["losses"]
    dynamic_astar_losses_array = dynamic_dict["losses"]

    # Find the overall maximum length across both datasets
    max_len = max(
        max((len(l) for l in static_astar_losses_array), default=0),
        max((len(l) for l in dynamic_astar_losses_array), default=0)
    )

    # padded numpy arrays
    static_astar_padded = pad_to_max(static_astar_losses_array, max_len)
    dynamic_astar_padded = pad_to_max(dynamic_astar_losses_array, max_len)

    # mean and std dev calculations with warnings ignored for NaN slices
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        
        static_astar_mean_loss = np.nanmean(static_astar_padded, axis=0)
        static_astar_std_loss = np.nanstd(static_astar_padded, axis=0)

        dynamic_astar_mean_loss = np.nanmean(dynamic_astar_padded, axis=0)
        dynamic_astar_std_loss = np.nanstd(dynamic_astar_padded, axis=0)

    labels = ["STATIC A-Star", "DYNAMIC A-Star"]

    static_astar_final_losses = np.array(static_dict["final_losses"])
    dynamic_astar_final_losses = np.array(dynamic_dict["final_losses"])



    # mean loss with standard deviation shading
    plot_mean_loss_with_std(labels, static_astar_mean_loss, static_astar_std_loss, dynamic_astar_mean_loss, dynamic_astar_std_loss, RUNS, f"{filename}.png", "California Housing")

    # box and whisker of final losses
    plot_final_loss_distribution(labels, static_astar_final_losses, dynamic_astar_final_losses, RUNS, f"{filename}.png", "California Housing")