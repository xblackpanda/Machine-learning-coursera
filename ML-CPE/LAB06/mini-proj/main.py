import json
import os

import numpy as np

from data_loader import load_dataset
from preprocessing import to_features
from nn_model import train_model, predict_model
from evaluate import evaluate_model, plot_history

# Paths are relative to this file, so the script runs from any directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

IMG_SIZE = 100
MAX_PER_CLASS = None   # None = use all images
EPOCHS = 50
BATCH_SIZE = 32


def main():

    print("--" * 30)
    print("Neural Network Image Recognition: coffee bean")
    print("--" * 30)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load Dataset
    print("\n[Step 1] Loading dataset...")
    splits, classes = load_dataset(DATA_PATH, IMG_SIZE, MAX_PER_CLASS)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["valid"]
    X_test, y_test = splits["test"]

    images = np.concatenate([X_train, X_val, X_test])
    labels = np.concatenate([y_train, y_val, y_test])

    np.save(f"{OUTPUT_DIR}/labels.npy", labels)
    with open(f"{OUTPUT_DIR}/classes.json", "w") as f:
        json.dump(classes, f)

    print("\nDataset loaded successfully.")
    print(f"Total images : {len(images)}")
    print(f"Classes      : {classes}")

    # Step 2: Preprocessing
    print("\n[Step 2] Preprocessing images...")

    X_train = to_features(X_train)
    X_val = to_features(X_val)
    X_test = to_features(X_test)

    np.save(f"{OUTPUT_DIR}/features.npy", np.concatenate(
        [X_train, X_val, X_test]
    ))

    print(f"Feature shape: {X_train.shape}")

    # Step 3: Use the train/valid/test split supplied with the dataset
    print("\n[Step 3] Using dataset splits...")

    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUTPUT_DIR}/X_val.npy", X_val)
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/y_val.npy", y_val)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

    print(f"Training samples  : {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Testing samples   : {len(X_test)}")

    # Step 4: Train Model
    print("\n[Step 4] Training model...")

    model, history = train_model(
        X_train, y_train, X_val, y_val, len(classes),
        OUTPUT_DIR, EPOCHS, BATCH_SIZE
    )

    print("Training completed.")

    # Step 5: Prediction
    print("\n[Step 5] Testing model...")
    predictions = predict_model(model, X_test)

    # Step 6: Evaluation
    print("\n[Step 6] Evaluating model...")
    evaluate_model(y_test, predictions, classes,
                   save_path=f"{OUTPUT_DIR}/confusion_matrix.png")
    plot_history(history, f"{OUTPUT_DIR}/training_history.png")


if __name__ == "__main__":
    main()
