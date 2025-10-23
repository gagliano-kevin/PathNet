import matplotlib.pyplot as plt


def plot_losses(loss_lists, labels, filename="loss_plot.png"):
    """
    Plots multiple lists of loss values on a single figure, regardless of length.

    Args:
        loss_lists (list of lists/arrays): The loss values for each training run.
        labels (list of str): Names for the legend of each run.
    """
    plt.figure(figsize=(10, 6))

    for i, loss_values in enumerate(loss_lists):
        # The X-axis for each list is simply its index, starting from 1 (iteration/epoch number)
        iterations = range(1, len(loss_values) + 1)

        # Plot the loss data
        plt.plot(iterations, loss_values, label=labels[i], alpha=0.8)

    # --- 3. Formatting the Plot ---
    plt.title('Comparison of Training Loss Histories')
    plt.xlabel('Iteration / Epoch Number')
    plt.ylabel('Loss Value')
    plt.legend(title='Type of Training')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout() # Adjusts plot to prevent labels from overlapping
    #plt.show()
    plt.savefig(filename)
    print(f"Training plot saved in file: {filename}")