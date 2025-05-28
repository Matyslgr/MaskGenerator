##
## EPITECH PROJECT, 2025
## MaskGenerator
## File description:
## plots
##

import matplotlib.pyplot as plt

def plot_folds_histories(histories_dict: dict, output_path: str):
    num_folds = len(histories_dict)
    fig, axes = plt.subplots(num_folds, 2, figsize=(14, 4 * num_folds))

    if num_folds == 1:
        axes = [axes]

    for i, (fold, history) in enumerate(histories_dict.items()):
        # Plot Losses (colonne de gauche)
        axes[i][0].plot(history["train_loss"], label="Train Loss")
        axes[i][0].plot(history["val_loss"], label="Val Loss")
        axes[i][0].set_title(f"Fold {fold} - Loss")
        axes[i][0].set_xlabel("Epoch")
        axes[i][0].set_ylabel("Loss")
        axes[i][0].legend()
        axes[i][0].grid(True)

        # Plot Metrics (colonne de droite)
        axes[i][1].plot(history["dice_score"], label="Dice Score")
        axes[i][1].plot(history["iou_score"], label="IOU Score")
        axes[i][1].set_title(f"Fold {fold} - Metrics")
        axes[i][1].set_xlabel("Epoch")
        axes[i][1].set_ylabel("Score")
        axes[i][1].legend()
        axes[i][1].grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"📊 Plots saved to {output_path}")

def plot_history(history: dict, output_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Losses (colonne de gauche)
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Plot Metrics (colonne de droite)
    axes[1].plot(history["dice_score"], label="Dice Score")
    axes[1].plot(history["iou_score"], label="IOU Score")
    axes[1].set_title("Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"📊 Plots saved to {output_path}")
