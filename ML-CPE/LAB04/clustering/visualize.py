import os
import pandas as pd
import matplotlib.pyplot as plt


def save_elbow_plot(wcss, output_path):
    """
    สร้างกราฟ Elbow Method
    """

    k_range = range(1, len(wcss) + 1)

    plt.figure(figsize=(8, 5))

    plt.plot(
        k_range,
        wcss,
        marker="o"
    )

    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("WCSS")

    plt.grid(True)

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()



def save_cluster_plot(X, labels, output_path):
    """
    แสดงผล Cluster ด้วย Feature 2 ตัวแรก
    """

    plt.figure(figsize=(8, 5))

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=labels,
        s=50
    )

    plt.title("K-Means Clustering Result")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    plt.grid(True)

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()



def save_cluster_data(df, labels, output_dir):
    """
    บันทึกผล Clustering เป็น CSV
    """

    result = df.copy()

    result["Cluster"] = labels


    # clustered_wine.csv

    clustered_file = os.path.join(
        output_dir,
        "clustered_wine.csv"
    )

    result.to_csv(
        clustered_file,
        index=False
    )


    # cluster_summary.csv

    summary = (
        result["Cluster"]
        .value_counts()
        .sort_index()
        .reset_index()
    )


    summary.columns = [
        "Cluster",
        "Number_of_Data"
    ]


    summary_file = os.path.join(
        output_dir,
        "cluster_summary.csv"
    )


    summary.to_csv(
        summary_file,
        index=False
    )