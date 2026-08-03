from sklearn.neighbors import KNeighborsClassifier


def train_knn(X_train, y_train, k):

    model = KNeighborsClassifier(n_neighbors=k)

    model.fit(X_train, y_train)

    return model