import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


def evaluate(model, X_test, y_test, k):

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    print(f"k = {k} Accuracy = {accuracy:.4f}")

    cm = confusion_matrix(y_test, prediction)

    disp = ConfusionMatrixDisplay(cm)

    disp.plot()

    plt.savefig("outputs/02_confusion_matrix.png")

    plt.close()

    result = pd.DataFrame({
        "Actual": y_test,
        "Prediction": prediction
    })

    result.to_csv("outputs/predictions.csv", index=False)

    return accuracy