import json
import numpy as np
import pandas as pd
from torchvision.datasets import OxfordIIITPet

# Load training and validation dataset
trainval_dataset = OxfordIIITPet(
    root="data/oxford_iiit_pet",
    split="trainval",
    target_types="category",
    download=False
)

# Load official test dataset
test_dataset = OxfordIIITPet(
    root="data/oxford_iiit_pet",
    split="test",
    target_types="category",
    download=False
)

# Load saved split indices
with open("data/splits/train_val_split.json", "r") as file:
    split_data = json.load(file)

train_indices = split_data["train_indices"]
val_indices = split_data["val_indices"]

# Get breed names and labels
breed_names = trainval_dataset.classes
trainval_labels = np.array(trainval_dataset._labels)
test_labels = np.array(test_dataset._labels)

# Count images for each breed
rows = []

for breed_id, breed_name in enumerate(breed_names):
    train_count = np.sum(trainval_labels[train_indices] == breed_id)
    val_count = np.sum(trainval_labels[val_indices] == breed_id)
    test_count = np.sum(test_labels == breed_id)

    rows.append({
        "Breed": breed_name,
        "Training": train_count,
        "Validation": val_count,
        "Test": test_count
    })

# Create and display table
df = pd.DataFrame(rows)

print("\nImages per breed:")
print(df.to_string(index=False))

# Save the counts
df.to_csv("results/metrics/breed_counts.csv", index=False)

print("\nBreed counts saved successfully!")