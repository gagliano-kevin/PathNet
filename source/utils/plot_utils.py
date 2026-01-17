import matplotlib.pyplot as plt
import numpy as np
import warnings
import json
import os



def pad_losses(losses_list, target_len):
    """Pads all loss histories in the list up to the target_len with NaN."""
    padded_array = np.full((len(losses_list), target_len), np.nan)
    for i, l in enumerate(losses_list):
        padded_array[i, :len(l)] = l
    return padded_array



# function to handle multiple training runs with varying lengths
def pad_to_max(list_of_lists, total_len):
    return np.array([
        l + [np.nan] * (total_len - len(l)) 
        for l in list_of_lists
    ])



def save_metrics(metrics_dict, filename):
    """
    Saves a dictionary of metrics to a JSON file.
    :param filename: String path to the file (e.g., 'results.json')
    :param metrics_dict: The dictionary containing your data
    """
    filename = filename + ".json"
    try:
        with open(filename, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        print(f"Successfully saved metrics to {filename}")
    except Exception as e:
        print(f"Error saving file: {e}")



def load_metrics(filename):
    """
    Loads metrics from a JSON file.
    :param filename: String path to the file
    :return: Dictionary containing the loaded data or None if failed
    """
    filename = filename + ".json"
    if not os.path.exists(filename):
        print(f"File {filename} not found.")
        return None
        
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded metrics from {filename}")
        return data
    except Exception as e:
        print(f"Error loading file: {e}")
        return None



def grad_astar_plot_mean_loss_with_std(astar_mean, astar_std, grad_mean, grad_std, runs, filename="mean_loss_comparison_with_std.png", dataset_name="dataset"):
    """Plots the mean loss over epochs/iterations with a shaded region for standard deviation.
        Method specifically for Gradient Descent vs A-Star comparison."""
    
    # epochs is now correctly determined by the global maximum length
    epochs = np.arange(len(astar_mean)) + 1
    
    plt.figure(figsize=(10, 6))

    # Plot A-Star 
    plt.plot(epochs, astar_mean, label='A-Star (Mean Loss)', color='blue')
    plt.fill_between(epochs, astar_mean - astar_std, astar_mean + astar_std, 
                     alpha=0.2, color='blue', label='A-Star ($\pm 1 \sigma$)')

    # Plot Gradient Descent 
    plt.plot(epochs, grad_mean, label='Gradient Descent (Mean Loss)', color='red')
    plt.fill_between(epochs, grad_mean - grad_std, grad_mean + grad_std, 
                     alpha=0.2, color='red', label='Gradient Descent ($\pm 1 \sigma$)')

    plt.title(f'Mean Training Loss Comparison on {dataset_name} over {runs} Runs')
    plt.xlabel('Epochs / Iterations')
    plt.ylabel('Mean Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot: {filename}")



def astar_plot_mean_loss_with_std(labels, static_astar_mean, static_astar_std, dynamic_astar_mean, dynamic_astar_std, runs, filename="mean_loss_comparison_with_std.png", dataset_name="dataset"):
    """Plots the mean loss over epochs/iterations with a shaded region for standard deviation.
        Method specifically for Static A-Star vs Dynamic A-Star comparison."""
    
    # get the maximum length for epochs
    epochs = np.arange(max(len(static_astar_mean), len(dynamic_astar_mean))) + 1
    
    plt.figure(figsize=(10, 6))

    # Plot Static A-Star
    plt.plot(epochs, static_astar_mean, label=f'{labels[0]} (Mean Loss)', color='blue')
    plt.fill_between(epochs, static_astar_mean - static_astar_std, static_astar_mean + static_astar_std, 
                     alpha=0.2, color='blue', label=f'{labels[0]} ($\pm 1 \sigma$)')

    # Plot Dynamic A-Star
    plt.plot(epochs, dynamic_astar_mean, label=f'{labels[1]}  (Mean Loss)', color='red')
    plt.fill_between(epochs, dynamic_astar_mean - dynamic_astar_std, dynamic_astar_mean + dynamic_astar_std, 
                     alpha=0.2, color='red', label=f'{labels[1]} ($\pm 1 \sigma$)')

    plt.title(f'Mean Training Loss Comparison on {dataset_name} over {runs} Runs')
    plt.xlabel('Epochs / Iterations')
    plt.ylabel('Mean Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot: {filename}")



def grad_astar_plot_final_loss_distribution(astar_final_losses, grad_final_losses, runs, filename="final_loss_boxplot.png", dataset_name="dataset"):
    """Plots a Box-and-Whisker plot of the final performance metric.
        Method specifically for Gradient Descent vs A-Star comparison."""
    
    data = [astar_final_losses, grad_final_losses]
    labels = ['A-Star', 'Gradient Descent']
    
    plt.figure(figsize=(8, 6))
    
    # Boxplot showing median, IQR, and range
    plt.boxplot(data, vert=True, patch_artist=True, labels=labels, 
                boxprops=dict(facecolor='lightblue'),
                medianprops=dict(color='darkred'))
    
    # Add individual points (jitter) to show all run results
    for i, losses in enumerate(data):
        x = np.random.normal(i + 1, 0.04, size=len(losses)) 
        plt.scatter(x, losses, color='black', alpha=0.6, s=10)

    plt.title(f'Distribution of Final Loss on {dataset_name} over {runs} Runs')
    plt.ylabel('Final Loss')
    plt.xticks(ticks=[1, 2], labels=labels)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot: {filename}")



def astar_plot_final_loss_distribution(labels, static_astar_final_losses, dynamic_astar_final_losses, runs, filename="final_loss_boxplot.png", dataset_name="dataset"):
    """Plots a Box-and-Whisker plot of the final performance metric.
        Method specifically for Static A-Star vs Dynamic A-Star comparison."""
    
    data = [static_astar_final_losses, dynamic_astar_final_losses]
    
    plt.figure(figsize=(8, 6))
    
    # Boxplot showing median, IQR, and range
    plt.boxplot(data, vert=True, patch_artist=True, labels=labels, 
                boxprops=dict(facecolor='lightblue'),
                medianprops=dict(color='darkred'))
    
    # Add individual points (jitter) to show all run results
    for i, losses in enumerate(data):
        x = np.random.normal(i + 1, 0.04, size=len(losses)) 
        plt.scatter(x, losses, color='black', alpha=0.6, s=10)

    plt.title(f'Distribution of Final Loss on {dataset_name} over {runs} Runs')
    plt.ylabel('Final Loss')
    plt.xticks(ticks=[1, 2], labels=labels)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot: {filename}")



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

    # check if dynamic_quantization_iterations and dynamic_kernel_reshaping_iterations exist in dynamic_dict
    if "dynamic_quantization_iterations" in dynamic_dict and "dynamic_kernel_reshaping_iterations" in dynamic_dict:
        # Print for each RUN the number of dynamic adjustments and the iteration when they occurred
        print("\nDynamic Adjustments per Run:")
        for run in range(RUNS):
            print(f" Run {run + 1}:")
            print(f"  - Dynamic Quantization Iterations: {dynamic_dict['dynamic_quantization_iterations'][run]}")
            print(f"  - Dynamic Kernel Reshaping Iterations: {dynamic_dict['dynamic_kernel_reshaping_iterations'][run]}")

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

        if "dynamic_quantization_iterations" in dynamic_dict and "dynamic_kernel_reshaping_iterations" in dynamic_dict:
            # Print for each RUN the number of dynamic adjustments and the iteration when they occurred
            f.write("\nDynamic Adjustments per Run:\n")
            for run in range(RUNS):
                f.write(f" Run {run + 1}:\n")
                f.write(f"  - Dynamic Quantization Iterations: {dynamic_dict['dynamic_quantization_iterations'][run]}\n")
                f.write(f"  - Dynamic Kernel Reshaping Iterations: {dynamic_dict['dynamic_kernel_reshaping_iterations'][run]}\n")

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
    astar_plot_mean_loss_with_std(labels, static_astar_mean_loss, static_astar_std_loss, dynamic_astar_mean_loss, dynamic_astar_std_loss, RUNS, f"{filename}_mean_loss.png", "California Housing")

    # box and whisker of final losses
    astar_plot_final_loss_distribution(labels, static_astar_final_losses, dynamic_astar_final_losses, RUNS, f"{filename}_final_loss.png", "California Housing")

    

