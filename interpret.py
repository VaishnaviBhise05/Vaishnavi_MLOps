from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Print feature importance
print("Feature Importance:")
for feature, importance in zip(iris.feature_names, model.feature_importances_):
    print(f"{feature}: {importance:.4f}")

# Plot feature importance
plt.figure(figsize=(7,5))
plt.bar(iris.feature_names, model.feature_importances_)
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=20)
plt.show()
