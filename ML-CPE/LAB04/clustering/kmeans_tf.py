from sklearn.cluster import KMeans


def train_kmeans(X, n_clusters=3):

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    model.fit(X)

    return model