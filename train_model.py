import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
X = np.load("X.npy")
y = np.load("y.npy")

# Create model
model = Sequential()

model.add(Dense(128, input_shape=(42,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))  # 3 classes: A, B, C

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(X, y, epochs=20)

# Save model
model.save("model.h5")

print("Model trained and saved!")