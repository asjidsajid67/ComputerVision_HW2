import os
import json
import random
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.datasets import OxfordIIITPet

# Set a fixed seed so we can reproduce the same split
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# Dataset location
data_dir = os.path.join("data", "oxford_iiit_pet")

# Load the official training and validation dataset
dataset = OxfordIIITPet(
    root=data_dir,
    split="trainval",
    target_types="category",
    download=False
)

# Get the breed labels for all images
labels = np.array(dataset._labels)

# Create training and validation indices
indices = np.arange(len(dataset))

train_indices, val_indices = train_test_split(
    indices,
    test_size=0.20,
    random_state=SEED,
    stratify=labels
)

# Create the output folder
os.makedirs("data/splits", exist_ok=True)

# Save the split indices
split_data = {
    "seed": SEED,
    "train_indices": train_indices.tolist(),
    "val_indices": val_indices.tolist()
}

with open("data/splits/train_val_split.json", "w") as file:
    json.dump(split_data, file, indent=4)

# Display split information
print("Dataset split completed!")
print("--------------------------------")
print("Total images:", len(dataset))
print("Training images:", len(train_indices))
print("Validation images:", len(val_indices))
print("Number of breeds:", len(np.unique(labels)))
print("--------------------------------")
print("Split file saved successfully.")