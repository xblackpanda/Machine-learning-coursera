from sklearn.cluster import KMeans


def calculate_wcss(X, max_k=10):

    wcss = []

    for k in range(1, max_k + 1):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(X)

        wcss.append(model.inertia_)

    return wcss