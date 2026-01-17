import matplotlib.pyplot as plt
import numpy as np
import warnings
import json
import os



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



def plot_mean_loss_with_std(labels, mean_losses, std_losses, runs, filename="mean_loss_comparison.png", dataset_name="dataset"):
    """
    Plots the mean loss with shaded standard deviation.
    Legend includes the sigma description but is placed outside to avoid overlap.
    """
    plt.figure(figsize=(12, 6)) 
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels))) 

    for i, (label, mean, std) in enumerate(zip(labels, mean_losses, std_losses)):
        curr_epochs = np.arange(len(mean)) + 1
        
        # Combined label for the legend
        combined_label = f'{label} (Mean $\pm 1 \sigma$)'
        
        # Plot the mean line with the descriptive label
        plt.plot(curr_epochs, mean, label=combined_label, color=colors[i], linewidth=2)
        
        # Plot the shading WITHOUT a label so it doesn't create a second legend entry
        plt.fill_between(curr_epochs, mean - std, mean + std, 
                         alpha=0.15, color=colors[i])

    plt.title(f'Mean Training Loss Comparison on {dataset_name} over {runs} Runs')
    plt.xlabel('Epochs / Iterations')
    plt.ylabel('Mean Loss')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Position the legend outside to the right
    # bbox_to_anchor=(1.02, 1) places it slightly to the right of the axes
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

    plt.tight_layout()
    # bbox_inches='tight' is critical here to ensure the external legend is saved
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {filename}")


def plot_final_loss_distribution(labels, final_losses_list, runs, filename="final_loss_boxplot.png", dataset_name="dataset"):
    """final_losses_list: list of np.arrays containing final losses for each run"""
    plt.figure(figsize=(8, 6))
    
    # Boxplot
    bplot = plt.boxplot(final_losses_list, vert=True, patch_artist=True, labels=labels, 
                        medianprops=dict(color='darkred'))
    
    # Aesthetic coloring
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(labels)))
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    
    # Individual Jitter Points
    for i, losses in enumerate(final_losses_list):
        x = np.random.normal(i + 1, 0.04, size=len(losses)) 
        plt.scatter(x, losses, color='black', alpha=0.5, s=12)

    plt.title(f'Distribution of Final Loss on {dataset_name} over {runs} Runs')
    plt.ylabel('Final Loss')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()



def generate_statistical_summary(data_dicts, labels, filename):
    RUNS = len(data_dicts[0]["final_losses"])
    standard_keys = ["losses", "final_losses", "training_times"]
    
    # 1. Determine dynamic column width for the main table
    max_label_len = max([len(l) for l in labels] + [15])
    col_width = max_label_len + 2 
    
    # 2. Header Construction
    header = "| Metric".ljust(20)
    separator = "-" * 19
    for label in labels:
        header += f"| {label}".ljust(col_width + 1)
        separator += "+" + ("-" * col_width)
    header += "|"

    metrics = ["Average Loss", "Median Loss", "Std Dev", "Min Loss", "Max Loss", "AVG Time"]
    
    lines = []
    total_table_width = len(header)
    lines.append("=" * total_table_width)
    lines.append(f" STATISTICAL SUMMARY over {RUNS} Runs ".center(total_table_width, "="))
    lines.append(header)
    lines.append(f"|{separator}|")

    # 3. Calculate core stats
    stats_collection = []
    for d in data_dicts:
        final = np.array(d["final_losses"])
        times = np.array(d["training_times"])
        stats_collection.append([
            np.mean(final), np.median(final), np.std(final), 
            np.min(final), np.max(final), np.mean(times)
        ])

    for i, metric in enumerate(metrics):
        row = f"| {metric}".ljust(20)
        for s in stats_collection:
            row += f"| {s[i]:.6f}".ljust(col_width + 1)
        lines.append(row + "|")
    lines.append("=" * total_table_width)

    # 4. Handle Dynamic/Extra Keys (Generic Implementation)
    extra_info_lines = []
    for d, label in zip(data_dicts, labels):
        # Identify keys that are not part of the standard metrics
        extra_keys = [k for k in d.keys() if k not in standard_keys]
        
        if extra_keys:
            extra_info_lines.append(f"\nExtra Details for {label}:")
            for run in range(RUNS):
                extra_info_lines.append(f" Run {run + 1}:")
                for k in extra_keys:
                    # Replace underscores with spaces for cleaner output
                    key_display = k.replace("_", " ").title()
                    val = d[k][run]
                    extra_info_lines.append(f"  - {key_display}: {val}")

    # Combine main table and extra info
    summary_text = "\n".join(lines + extra_info_lines)
    
    # Print and Save
    print(summary_text)
    with open(f"{filename}.txt", "w") as f:
        f.write(summary_text)
        
    print(f"\nSaved statistical summary to '{filename}.txt'")



def generate_plots(data_dicts, labels, filename, dataset_name="California Housing"):
    """
    data_dicts: List of dictionaries (e.g. [static_dict, dynamic_dict, custom_dict])
    labels: List of strings (e.g. ["Static", "Dynamic", "Optimized"])
    """
    RUNS = len(data_dicts[0]["final_losses"])
    
    # Calculate Max Length for padding
    all_loss_sequences = [d["losses"] for d in data_dicts]
    max_len = 0
    for seq_list in all_loss_sequences:
        for seq in seq_list:
            max_len = max(max_len, len(seq))

    # Calculate Mean and Std for all
    mean_list, std_list, final_list = [], [], []
    
    for d in data_dicts:
        padded = pad_to_max(d["losses"], max_len)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_list.append(np.nanmean(padded, axis=0))
            std_list.append(np.nanstd(padded, axis=0))
        final_list.append(np.array(d["final_losses"]))

    # Call plotting functions
    plot_mean_loss_with_std(labels, mean_list, std_list, RUNS, f"{filename}_mean_loss.png", dataset_name)
    plot_final_loss_distribution(labels, final_list, RUNS, f"{filename}_final_loss.png", dataset_name)


"""
Helpers for formatting numbers in scientific notation for plot labels and summaries.
"""
def format_sci(value):
    """
    Converts a number to scientific notation (e.g., 100.0 -> 1e2).
    - .0e: scientific notation with 0 decimal points
    - replace('+', ''): removes the plus sign
    - replace('e0', 'e'): handles '1e+02' -> '1e2' (removes leading zero in exponent)
    - replace('.0', ''): safety to ensure no decimal point remains
    """
    s = f"{value:.0e}".replace("+", "").replace(".0", "")
    return s.replace("e0", "e") if "e0" in s else s