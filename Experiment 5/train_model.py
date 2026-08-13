from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load Iris dataset
iris = load_iris()

# Features and target
X = iris.data
y = iris.target

# Create Random Forest model
model = RandomForestClassifier(random_state=42)

# Train model
model.fit(X, y)

# Create model directory
os.makedirs("model", exist_ok=True)

# Save trained model
joblib.dump(model, "model/model.pkl")

print("Model trained successfully.")
print("Model saved at: model/model.pkl")