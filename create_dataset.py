import numpy as np
import os

data = []
labels = []

alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

for idx, label in enumerate(alphabet):
    path = f"dataset/{label}"

    # 🔥 Skip folder if it does not exist
    if not os.path.exists(path):
        print(f"Skipping {label} (folder not found)")
        continue

    for file in os.listdir(path):
        file_path = os.path.join(path, file)

        try:
            arr = np.load(file_path)
            data.append(arr)
            labels.append(idx)
        except:
            print(f"Error reading file: {file_path}")

# Convert to numpy arrays
X = np.array(data)
y = np.array(labels)

# Save dataset
np.save("X.npy", X)
np.save("y.npy", y)

print("Dataset created successfully!")
print("X shape:", X.shape)
print("y shape:", y.shape)
