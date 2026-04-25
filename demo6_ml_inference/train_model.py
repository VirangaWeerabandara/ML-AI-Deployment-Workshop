import pickle
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Save model
os.makedirs("models", exist_ok=True)
with open("models/iris_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved to models/iris_model.pkl")

# Save feature names for reference
with open("models/feature_names.pkl", "wb") as f:
    pickle.dump(iris.feature_names, f)

with open("models/target_names.pkl", "wb") as f:
    pickle.dump(iris.target_names, f)

print("Feature and target names saved!")
