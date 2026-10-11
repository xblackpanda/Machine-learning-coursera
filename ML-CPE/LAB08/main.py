import json
import os

import numpy as np

from data_loader import load_data
from preprocessing import to_features
from split_data import split_dataset
from vgg_model import (
    train_model,
    predict_model,
    VGG16,
    VGG_SMALL,
    VGG_MEDIUM,
    VGG_TINY,
)
from evaluate import evaluate_model, plot_history, plot_model_comparison

# Paths are relative to this file, so the script runs from any directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DATA_PATH = os.path.join(DATASET_DIR, "entrainement")
TEST_DATA_PATH = os.path.join(DATASET_DIR, "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

IMG_SIZE = 50
VAL_SIZE = 0.2
MAX_PER_CLASS = None   # None = use all images
EPOCHS = 20
BATCH_SIZE = 64

MODEL_CONFIGS = {
    "vgg16": VGG16,
    "vgg_small": VGG_SMALL,
    "vgg_medium": VGG_MEDIUM,
    "vgg_tiny": VGG_TINY,
}


def main():

    print("--" * 30)
    print("Waste Classification: Multiple VGG Models")
    print("--" * 30)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load Dataset
    print("\n[Step 1] Loading dataset...")
    images, labels, classes = load_data(
        TRAIN_DATA_PATH, IMG_SIZE, MAX_PER_CLASS
    )
    test_images, test_labels, _ = load_data(
        TEST_DATA_PATH, IMG_SIZE, class_names=classes
    )

    np.save(f"{OUTPUT_DIR}/labels.npy", labels)
    with open(f"{OUTPUT_DIR}/classes.json", "w") as f:
        json.dump(classes, f)

    print("\nDataset loaded successfully.")
    print(f"Total images : {len(images)}")
    print(f"Classes      : {classes}")

    # Step 2: Preprocessing
    print("\n[Step 2] Preprocessing images...")

    X = to_features(images)
    y = labels
    X_test = to_features(test_images)
    y_test = test_labels

    np.save(f"{OUTPUT_DIR}/features.npy", X)

    print(f"Feature shape: {X.shape}")

    # Step 3: Split Dataset
    print("\n[Step 3] Splitting dataset...")

    X_train, X_val, _, y_train, y_val, _ = split_dataset(
        X, y, 0, VAL_SIZE
    )

    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUTPUT_DIR}/X_val.npy", X_val)
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/y_val.npy", y_val)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

    print(f"Training samples  : {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Testing samples   : {len(X_test)}")

    print(f"Test samples      : {len(X_test)}")

    # Train and evaluate each architecture on the same data split.
    metrics = {}
    for model_name, blocks in MODEL_CONFIGS.items():
        model_output_dir = os.path.join(OUTPUT_DIR, model_name)
        print(f"\n[Step 4] Training {model_name}...")
        model, history = train_model(
            X_train, y_train, X_val, y_val, len(classes),
            model_output_dir, EPOCHS, BATCH_SIZE, blocks
        )

        print(f"\n[Step 5] Evaluating {model_name} on the test dataset...")
        predictions = predict_model(model, X_test)
        metrics[model_name] = float(evaluate_model(
            y_test,
            predictions,
            classes,
            save_path=os.path.join(model_output_dir, "confusion_matrix.png"),
        ))
        plot_history(
            history, os.path.join(model_output_dir, "training_history.png")
        )

    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nModel accuracies saved to {os.path.join(OUTPUT_DIR, 'metrics.json')}")
    plot_model_comparison(
        metrics, os.path.join(OUTPUT_DIR, "model_comparison.png")
    )


if __name__ == "__main__":
    main()
