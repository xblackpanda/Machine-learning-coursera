import os

from data_loader import load_data
from kmeans_tf import train_kmeans
from knn_tools import calculate_wcss

from visualize import (
    save_elbow_plot,
    save_cluster_plot,
    save_cluster_data
)


# =========================
# Path Setup
# =========================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


DATA_PATH = os.path.join(
    BASE_DIR,
    "..",
    "data-wine",
    "Wine dataset.csv"
)


OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)



# =========================
# Load Dataset
# =========================

print("Loading dataset...")

X_scaled, y, df = load_data(
    DATA_PATH
)


print("Dataset loaded")
print(
    "Data size:",
    X_scaled.shape
)



# =========================
# Elbow Method
# =========================

print("\nCalculating Elbow Method...")

wcss = calculate_wcss(
    X_scaled,
    max_k=10
)


elbow_path = os.path.join(
    OUTPUT_DIR,
    "01_elbow.png"
)


save_elbow_plot(
    wcss,
    elbow_path
)


print(
    "Saved:",
    elbow_path
)



# =========================
# K-Means Training
# =========================

print("\nTraining K-Means...")


kmeans = train_kmeans(
    X_scaled,
    n_clusters=3
)


labels = kmeans.labels_


print("Cluster completed")



# =========================
# Visualization
# =========================


cluster_path = os.path.join(
    OUTPUT_DIR,
    "02_clusters.png"
)


save_cluster_plot(
    X_scaled,
    labels,
    cluster_path
)


print(
    "Saved:",
    cluster_path
)



# =========================
# Save CSV
# =========================

save_cluster_data(
    df,
    labels,
    OUTPUT_DIR
)


print("\nCSV files created")


