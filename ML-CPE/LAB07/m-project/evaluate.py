import matplotlib

# Set backend before pyplot, so it works without a display
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def evaluate_model(y_test, predictions, classes, save_path=None):

    # Pin label order so target_names always matches the columns
    labels = list(range(len(classes)))

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    print("\n------------ Evaluation ------------------")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")

    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        target_names=classes,
        zero_division=0
    )

    print(report)
    print("Confusion Matrix:")

    matrix = confusion_matrix(y_test, predictions, labels=labels)
    print(matrix)

    if save_path:
        plot_confusion_matrix(matrix, classes, save_path)
        print(f"Saved: {save_path}")

    return accuracy


def plot_confusion_matrix(matrix, classes, save_path):

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(classes)), classes)
    ax.set_yticks(np.arange(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (%)")

    threshold = matrix.max() / 2
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, matrix[i, j], ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_history(history, save_path):
    """
    สร้างกราฟประเมินผลแบบ 2x2 Grid ตามสไตล์งานวิจัย IEEE
    (a) Training Accuracy  (b) Validation Accuracy
    (c) Training Loss      (d) Validation Loss
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    epochs = range(1, len(history.history["accuracy"]) + 1)

    # (a) Training Accuracy of model
    axes[0, 0].plot(epochs, history.history["accuracy"], color="purple", label="Proposed CNN")
    axes[0, 0].set_xlabel("Epochs")
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].set_title("(a) Training accuracy of model")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)
    axes[0, 0].legend(loc="lower right")

    # (b) Validation Accuracy of model
    axes[0, 1].plot(epochs, history.history["val_accuracy"], color="purple", label="Proposed CNN")
    axes[0, 1].set_xlabel("Epochs")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].set_title("(b) Validation accuracy of model")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)
    axes[0, 1].legend(loc="lower right")

    # (c) Training Loss of model
    axes[1, 0].plot(epochs, history.history["loss"], color="purple", label="Proposed CNN")
    axes[1, 0].set_xlabel("Epochs")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].set_title("(c) Training loss of model")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)
    axes[1, 0].legend(loc="upper right")

    # (d) Validation Loss of model
    axes[1, 1].plot(epochs, history.history["val_loss"], color="purple", label="Proposed CNN")
    axes[1, 1].set_xlabel("Epochs")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].set_title("(d) Validation loss of model")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)
    axes[1, 1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {save_path}")
