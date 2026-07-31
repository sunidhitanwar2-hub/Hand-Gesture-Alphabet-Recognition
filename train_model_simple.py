import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Load data
X = np.load("X.npy")
y = np.load("y.npy")

# Train model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("Model trained and saved!")