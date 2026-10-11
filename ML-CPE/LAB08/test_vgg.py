import json
import os
from glob import glob

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

N_SAMPLES = 4


def test_vgg(n_samples=N_SAMPLES):

    model_paths = sorted(glob(os.path.join(OUTPUT_DIR, "*", "model.keras")))
    legacy_model_path = os.path.join(OUTPUT_DIR, "vgg_model.keras")
    if not model_paths and os.path.isfile(legacy_model_path):
        model_paths = [legacy_model_path]
    if not model_paths:
        raise FileNotFoundError(
            f"No trained models found under {os.path.abspath(OUTPUT_DIR)}"
        )

    X_test = np.load(f"{OUTPUT_DIR}/X_test.npy")
    y_test = np.load(f"{OUTPUT_DIR}/y_test.npy")
    with open(f"{OUTPUT_DIR}/classes.json") as f:
        classes = json.load(f)

    if n_samples > len(X_test):
        raise ValueError(f"Requested {n_samples} samples, but only {len(X_test)} exist")

    index = np.random.choice(len(X_test), n_samples, replace=False)
    X_sample = X_test[index]
    y_sample = y_test[index]

    for model_path in model_paths:
        model_name = os.path.basename(os.path.dirname(model_path))
        if model_path == legacy_model_path:
            model_name = "vgg_model"
        model = keras.models.load_model(model_path)

        probabilities = model.predict(X_sample, verbose=0)
        if probabilities.shape[-1] == 1:
            probabilities = probabilities.ravel()
            predictions = (probabilities > 0.5).astype(int)
            confidence = np.where(predictions == 1, probabilities, 1 - probabilities)
        else:
            predictions = probabilities.argmax(axis=1)
            confidence = probabilities.max(axis=1)

        cols = int(np.ceil(np.sqrt(n_samples)))
        rows = int(np.ceil(n_samples / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 4.0 * rows))
        axes = np.atleast_1d(axes).ravel()

        for i, ax in enumerate(axes):
            if i >= n_samples:
                ax.axis("off")
                continue

            pred = classes[predictions[i]]
            true = classes[y_sample[i]]
            correct = predictions[i] == y_sample[i]
            color = "green" if correct else "red"

            ax.imshow(X_sample[i])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"Pred: {pred} ({confidence[i] * 100:.0f}%)\n"
                         f"True: {true}", color=color)

            print(f"{model_name} [{i + 1}] Pred: {pred}  True: {true}  "
                  f"conf {confidence[i] * 100:5.1f}%  "
                  f"{'OK' if correct else 'WRONG'}")

        correct_total = int((predictions == y_sample).sum())
        print(f"{model_name}: {correct_total}/{n_samples} correct")

        fig.suptitle(f"{model_name}: {correct_total}/{n_samples} correct")
        fig.tight_layout()

        save_dir = os.path.dirname(model_path)
        save_path = os.path.join(save_dir, "prediction_sample.png")
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    test_vgg()
