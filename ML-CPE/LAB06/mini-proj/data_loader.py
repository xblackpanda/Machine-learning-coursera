import csv
import os

import cv2
import numpy as np

from preprocessing import preprocess_image

VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def _load_split(split_path, img_size=100, max_per_class=None):
    csv_path = os.path.join(split_path, "_classes.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Missing label file: {csv_path}")

    images = []
    labels = []
    class_counts = None

    with open(csv_path, newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file, skipinitialspace=True)
        if not reader.fieldnames or "filename" not in reader.fieldnames:
            raise ValueError(f"{csv_path} must contain a filename column")

        classes = [name for name in reader.fieldnames if name != "filename"]
        class_counts = [0] * len(classes)

        for row in reader:
            filename = row["filename"].strip()
            values = [int(row[name]) for name in classes]
            if sum(values) != 1:
                continue

            label = values.index(1)
            if max_per_class is not None and class_counts[label] >= max_per_class:
                continue

            image = preprocess_image(
                cv2.imread(os.path.join(split_path, filename)), img_size
            )
            if image is None:
                continue

            images.append(image)
            labels.append(label)
            class_counts[label] += 1

    if not images:
        raise ValueError(f"No readable labelled images found in {split_path}")

    return np.stack(images), np.asarray(labels, dtype=np.int64), classes


def load_dataset(data_path, img_size=100, max_per_class=None):
    """Load the train/valid/test split exported by Roboflow."""

    splits = {}
    classes = None
    for split_name in ("train", "valid", "test"):
        split_path = os.path.join(data_path, split_name)
        split_data = _load_split(split_path, img_size, max_per_class)
        if classes is None:
            classes = split_data[2]
        elif split_data[2] != classes:
            raise ValueError(f"Class columns differ in {split_path}")
        splits[split_name] = split_data[:2]
        print(f"Loaded {split_name}: {len(split_data[0])} images")

    return splits, classes


def load_data(data_path, img_size=100, max_per_class=None):
    """Load a class-folder dataset, or combine a Roboflow dataset."""

    if all(os.path.isdir(os.path.join(data_path, split))
           for split in ("train", "valid", "test")):
        splits, classes = load_dataset(data_path, img_size, max_per_class)
        images = np.concatenate([splits[name][0]
                                 for name in ("train", "valid", "test")])
        labels = np.concatenate([splits[name][1]
                                 for name in ("train", "valid", "test")])
        return images, labels, classes

    images = []
    labels = []

    # Detect classes automatically from subdirectories
    classes = sorted([
        folder
        for folder in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, folder))
    ])
    print("Detected classes:", classes)

    # Read images from each class directory
    for label, class_name in enumerate(classes):
        class_path = os.path.join(data_path, class_name)
        filenames = sorted(
            f for f in os.listdir(class_path)
            if f.lower().endswith(VALID_EXT)
        )

        loaded = 0
        skipped = 0
        for filename in filenames:
            if max_per_class and loaded >= max_per_class:
                break

            image_path = os.path.join(class_path, filename)
            image = cv2.imread(image_path)

            # Resize here so full-size images are not all kept in memory
            image = preprocess_image(image, img_size)

            # Skip unreadable or damaged images
            if image is None:
                skipped += 1
                continue

            images.append(image)
            labels.append(label)
            loaded += 1

        print(f"Loaded class {class_name}: {loaded} images ({skipped} skipped)")

    return np.stack(images), np.array(labels), classes
