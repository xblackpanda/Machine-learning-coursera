import os
import matplotlib.pyplot as plt
from pathlib import Path

from data_loader import load_data
from knn_model import train_knn
from evaluate import evaluate
from pathlib import Path

os.makedirs("outputs", exist_ok=True)



BASE_DIR = Path(__file__).resolve().parent
DATASET = BASE_DIR.parent / "data-wine" / "Wine dataset.csv"

X_train, X_test, y_train, y_test = load_data(DATASET)


k_values = [3, 5, 7]

accuracies = []

best_k = 0
best_accuracy = 0

for k in k_values:

    model = train_knn(X_train, y_train, k)

    accuracy = evaluate(model, X_test, y_test, k)

    accuracies.append(accuracy)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

# วาดกราฟ Accuracy vs k
plt.figure(figsize=(6,4))
plt.plot(k_values, accuracies, marker='o')
plt.title("KNN Accuracy vs k")
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.xticks(k_values)
plt.grid(True)

plt.savefig("outputs/01_k_curve.png")
plt.close()

print("\n========== RESULT ==========")
print(f"Best k : {best_k}")
print(f"Accuracy : {best_accuracy:.4f}")