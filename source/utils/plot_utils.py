import matplotlib.pyplot as plt
import numpy as np
import warnings
import json
import os
import seaborn as sns



def pad_to_max(list_of_lists, total_len):
    return np.array([
        l + [np.nan] * (total_len - len(l)) 
        for l in list_of_lists
    ])



def save_metrics(metrics_dict, filename):
    """
    Saves a dictionary of metrics to a JSON file inside the specified directory.
    Handles non-serializable types like float32, int64, and numpy arrays.
    """
    os.makedirs(filename, exist_ok=True)
    file_path = os.path.join(filename, f"{filename}.json")

    #how to handle types it doesn't recognize
    def default_encoder(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj) # Fallback to string

    try:
        with open(file_path, 'w') as f:
            json.dump(metrics_dict, f, indent=4, default=default_encoder)
        print(f"Successfully saved metrics to {file_path}")
    except Exception as e:
        print(f"Error saving file: {e}")



def load_metrics(filename):
    """
    Loads metrics from a JSON file inside the specified directory.
    :param filename: Directory name
    :return: Dictionary containing the loaded data or None if failed
    """
    file_path = os.path.join(filename, f"{filename}.json")
    
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return None
        
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded metrics from {file_path}")
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
        
        combined_label = f'{label} (Mean $\pm 1 \sigma$)'
        
        plt.plot(curr_epochs, mean, label=combined_label, color=colors[i], linewidth=2)
        
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
    # external legend
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {filename}")


def plot_final_loss_distribution(labels, final_losses_list, runs, filename="final_loss_boxplot.png", dataset_name="dataset"):
    """final_losses_list: list of np.arrays containing final losses for each run"""
    plt.figure(figsize=(8, 6))
    
    bplot = plt.boxplot(final_losses_list, vert=True, patch_artist=True, labels=labels, 
                        medianprops=dict(color='darkred'))
    
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(labels)))
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    
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
    
    # dynamic column width for the main table
    max_label_len = max([len(l) for l in labels] + [15])
    col_width = max_label_len + 2 
    
    # header
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

    # core stats
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

    # handle extra keys
    extra_info_lines = []
    for d, label in zip(data_dicts, labels):
        extra_keys = [k for k in d.keys() if k not in standard_keys]
        
        if extra_keys:
            extra_info_lines.append(f"\nExtra Details for {label}:")
            for run in range(RUNS):
                extra_info_lines.append(f" Run {run + 1}:")
                for k in extra_keys:
                    key_display = k.replace("_", " ").title()
                    val = d[k][run]
                    extra_info_lines.append(f"  - {key_display}: {val}")

    summary_text = "\n".join(lines + extra_info_lines)

    os.makedirs(filename, exist_ok=True)

    print(summary_text)

    save_path = os.path.join(filename, f"{filename}_stats_summary.txt")

    with open(save_path, "w") as f:
        f.write(summary_text)
        
    print(f"\nSaved statistical summary to '{save_path}'")



def generate_plots(data_dicts, labels, filename, dataset_name="California Housing"):
    """
    data_dicts: List of dictionaries (e.g. [static_dict, dynamic_dict, custom_dict])
    labels: List of strings (e.g. ["Static", "Dynamic", "Optimized"])
    """
    RUNS = len(data_dicts[0]["final_losses"])
    
    # calculate max length for padding
    all_loss_sequences = [d["losses"] for d in data_dicts]
    max_len = 0
    for seq_list in all_loss_sequences:
        for seq in seq_list:
            max_len = max(max_len, len(seq))

    mean_list, std_list, final_list = [], [], []
    
    for d in data_dicts:
        padded = pad_to_max(d["losses"], max_len)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_list.append(np.nanmean(padded, axis=0))
            std_list.append(np.nanstd(padded, axis=0))
        final_list.append(np.array(d["final_losses"]))

    os.makedirs(filename, exist_ok=True)

    mean_loss_path = os.path.join(filename, f"{filename}_mean_loss.png")
    final_loss_path = os.path.join(filename, f"{filename}_final_loss.png")

    plot_mean_loss_with_std(labels, mean_list, std_list, RUNS, mean_loss_path, dataset_name)
    plot_final_loss_distribution(labels, final_list, RUNS, final_loss_path, dataset_name)



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



def generate_regression_statistical_summary(data_dicts, labels, filename):
    RUNS = len(data_dicts[0]["final_losses"])
    standard_keys = ["losses", "final_losses", "training_times", "evaluation_scores"]
    
    max_label_len = max([len(l) for l in labels] + [15])
    col_width = max_label_len + 2 
    
    header = "| Metric".ljust(25)
    separator = "-" * 24
    for label in labels:
        header += f"| {label}".ljust(col_width + 1)
        separator += "+" + ("-" * col_width)
    header += "|"

    eval_keys = []
    if "evaluation_scores" in data_dicts[0] and len(data_dicts[0]["evaluation_scores"]) > 0:
        eval_keys = list(data_dicts[0]["evaluation_scores"][0].keys())

    display_metrics = [
        "Avg Final Loss", "Median Final Loss", "Std Final Loss", 
        "Var Final Loss", "Min Final Loss", "Max Final Loss", 
        "Avg Time"
    ]
    for k in eval_keys:
        display_metrics.extend([f"Avg {k}", f"Std {k}", f"Min {k}", f"Max {k}"])
    
    lines = []
    total_table_width = len(header)
    lines.append("=" * total_table_width)
    lines.append(f" STATISTICAL SUMMARY over {RUNS} Runs ".center(total_table_width, "="))
    lines.append(header)
    lines.append(f"|{separator}|")

    stats_map = [] 
    for d in data_dicts:
        label_stats = {}
        final = np.array(d["final_losses"])
        times = np.array(d["training_times"])
        
        label_stats["Avg Final Loss"] = np.mean(final)
        label_stats["Median Final Loss"] = np.median(final)
        label_stats["Std Final Loss"] = np.std(final)
        label_stats["Var Final Loss"] = np.var(final)
        label_stats["Min Final Loss"] = np.min(final)
        label_stats["Max Final Loss"] = np.max(final)
        label_stats["Avg Time"] = np.mean(times)
        
        # Evaluation Score Stats
        if eval_keys:
            for k in eval_keys:
                scores = np.array([run_score[k] for run_score in d["evaluation_scores"]])
                label_stats[f"Avg {k}"] = np.mean(scores)
                label_stats[f"Std {k}"] = np.std(scores)
                label_stats[f"Min {k}"] = np.min(scores)
                label_stats[f"Max {k}"] = np.max(scores)
        
        stats_map.append(label_stats)

    # Build Table Rows with Group Separation
    for i, metric_name in enumerate(display_metrics):
        row = f"| {metric_name}".ljust(25)
        for s in stats_map:
            val = s.get(metric_name, 0.0)
            row += f"| {val:.6f}".ljust(col_width + 1)
        lines.append(row + "|")
        
        # Logic for visual separators:
        # After "Max Final Loss" (index 5)
        if i == 5:
            lines.append(f"|{separator}|")
        # After "Avg Time" (index 6)
        elif i == 6:
            lines.append(f"|{separator}|")
        # Between Eval Score groups (every 4 metrics after index 6)
        elif i > 6 and (i - 6) % 4 == 0 and i != len(display_metrics) - 1:
            lines.append(f"|{separator}|")

    lines.append("=" * total_table_width)

    # Extra Details (Dynamic adjustments like Quantization changes)
    extra_info_lines = []
    for d, label in zip(data_dicts, labels):
        extra_keys = [k for k in d.keys() if k not in standard_keys]
        if extra_keys:
            extra_info_lines.append(f"\nExtra Details for {label}:")
            for run in range(RUNS):
                run_details = [f"{k.replace('_', ' ').title()}: {d[k][run]}" for k in extra_keys]
                extra_info_lines.append(f" Run {run + 1}: " + " | ".join(run_details))

    # Save logic
    summary_text = "\n".join(lines + extra_info_lines)
    os.makedirs(filename, exist_ok=True)
    save_path = os.path.join(filename, f"{filename}_stats_summary.txt")
    
    with open(save_path, "w") as f:
        f.write(summary_text)
    
    print(summary_text)
    print(f"\nSaved statistical summary to '{save_path}'")



def plot_regression_statistics(data_dicts, labels, filename, dataset_name="Dataset"):
    """
    Generates a comprehensive visual summary of regression performance.
    Specifically, it creates:
    1. Boxplots for evaluation metrics (MSE, RMSE, R2, MAE).
    2. Mean loss convergence plot with standard deviation shading.
    """
    os.makedirs(filename, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # Boxplots for Evaluation Metrics (MSE, RMSE, R2, MAE)
    eval_keys = list(data_dicts[0]["evaluation_scores"][0].keys())
    num_metrics = len(eval_keys)
    
    fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 6))
    if num_metrics == 1: axes = [axes]

    for i, metric in enumerate(eval_keys):
        plot_data = []
        plot_labels = []
        
        for d, label in zip(data_dicts, labels):
            scores = [run[metric] for run in d["evaluation_scores"]]
            plot_data.extend(scores)
            plot_labels.extend([label] * len(scores))
        
        sns.boxplot(
            x=plot_labels, 
            y=plot_data, 
            ax=axes[i], 
            hue=plot_labels,    
            palette="viridis", 
            legend=False
        )

        axes[i].set_title(f"Distribution of {metric}")
        axes[i].set_ylabel("Score Value")
        axes[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(filename, f"{filename}_metrics_distribution.png"))
    plt.close()

    # Mean Loss Convergence Plot (with Standard Deviation Shadow)
    plt.figure(figsize=(10, 6))
    for d, label in zip(data_dicts, labels):
        losses = np.array(d["losses"])
        mean_loss = np.mean(losses, axis=0)
        std_loss = np.std(losses, axis=0)
        iters = np.arange(len(mean_loss))
        
        line, = plt.plot(iters, mean_loss, label=label, lw=2)
        plt.fill_between(iters, mean_loss - std_loss, mean_loss + std_loss, 
                         color=line.get_color(), alpha=0.2)

    plt.yscale('log')
    plt.title(f"Training Convergence on {dataset_name}")
    plt.xlabel("Iterations")
    plt.ylabel("Loss (Log Scale)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig(os.path.join(filename, f"{filename}_convergence.png"))
    plt.close()