
import os
from torchvision.datasets import OxfordIIITPet

# Folder where the dataset will be stored
data_dir = os.path.join("data", "oxford_iiit_pet")

# Create the folder if it does not already exist
os.makedirs(data_dir, exist_ok=True)

print("Starting dataset download...")

# Load the official training and validation dataset
trainval_dataset = OxfordIIITPet(
    root=data_dir,
    split="trainval",
    target_types="category",
    download=True
)

# Load the official test dataset
test_dataset = OxfordIIITPet(
    root=data_dir,
    split="test",
    target_types="category",
    download=True
)

# Count the number of breed categories
number_of_breeds = len(set(trainval_dataset._labels))

# Display basic information about the downloaded dataset
print("\nDataset download completed.")
print("--------------------------------")
print("Training and validation images:", len(trainval_dataset))
print("Test images:", len(test_dataset))
print("Total breed categories:", number_of_breeds)
print("--------------------------------")
