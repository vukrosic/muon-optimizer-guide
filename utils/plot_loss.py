import json
import matplotlib.pyplot as plt
import os

def plot_loss(metrics_file: str, plot_file: str, title: str = "Validation Loss", baseline_file: str = None):
    with open(metrics_file, "r") as f:
        data = json.load(f)

    epochs = data["history"]["steps"]
    val_loss = data["history"]["val_losses"]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_loss, marker="o", linestyle="-", label="Current")

    if baseline_file and os.path.exists(baseline_file):
        with open(baseline_file, "r") as f:
            baseline_data = json.load(f)
        b_epochs = baseline_data["history"]["steps"]
        b_val_loss = baseline_data["history"]["val_losses"]
        plt.plot(b_epochs, b_val_loss, marker="o", linestyle="--", label="Baseline")

    plt.title(title)
    plt.xlabel("Steps")
    plt.ylabel("Validation Loss")
    plt.grid(True)
    plt.legend()

    plt.savefig(plot_file)
    plt.close()
