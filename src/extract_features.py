
import os
import pickle
import json
import random
import time

import numpy as np
import torch
import torchvision

from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader, Subset

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix



# --------------------------------------------------
# STEP 2: Set random seeds and device
# --------------------------------------------------

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)



# --------------------------------------------------
# STEP 3: Set dataset and split file paths
# --------------------------------------------------

# Main folder containing the Oxford-IIIT Pet dataset
data_dir = "data/oxford_iiit_pet"

# File containing our training and validation split
split_file = "data/splits/train_val_split.json"

# Load the saved training and validation indices
with open(split_file, "r") as file:
    split_data = json.load(file)

# Get the indices for training and validation images
train_indices = split_data["train_indices"]
val_indices = split_data["val_indices"]

print("Dataset paths are ready.")
print("Training images:", len(train_indices))
print("Validation images:", len(val_indices))



# --------------------------------------------------
# STEP 4: Load pretrained MobileNetV3 weights
# --------------------------------------------------

# Load the default ImageNet pretrained weights
weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT

# Use the normalization settings required by these weights
image_transforms = weights.transforms()

print("Pretrained MobileNetV3 weights loaded.")
print("Image transformations are ready.")



# --------------------------------------------------
# STEP 5: Load the Oxford-IIIT Pet dataset
# --------------------------------------------------

# Load the official training and validation dataset
train_dataset = OxfordIIITPet(
    root=data_dir,
    split="trainval",
    target_types="category",
    transform=image_transforms,
    download=False
)

# Load the same dataset for validation
val_dataset = OxfordIIITPet(
    root=data_dir,
    split="trainval",
    target_types="category",
    transform=image_transforms,
    download=False
)
# Load the official test dataset
test_dataset = OxfordIIITPet(
    root=data_dir,
    split="test",
    target_types="category",
    transform=image_transforms,
    download=False
)

print("Test images:", len(test_dataset))
# Select the same training and validation images used earlier
train_dataset = Subset(train_dataset, train_indices)
val_dataset = Subset(val_dataset, val_indices)

print("Datasets loaded successfully.")
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))



# --------------------------------------------------
# STEP 6: Create DataLoaders
# --------------------------------------------------

# Use batches of 32 images
batch_size = 32

# Load training images in batches
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

# Load validation images in batches
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)
# Load test images in batches
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)
print("DataLoaders created successfully.")
print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))
print("Test batches:", len(test_loader))


# --------------------------------------------------
# STEP 7: Create the feature extraction model
# --------------------------------------------------

# Load MobileNetV3 with pretrained ImageNet weights
model = torchvision.models.mobilenet_v3_small(weights=weights)

# Remove the final classifier layers
# We only want the image features, not the 1000 ImageNet predictions
model.classifier = torch.nn.Identity()

# Move the model to the selected device
model = model.to(device)

# Freeze all model parameters
# This prevents the pretrained network from being changed
for parameter in model.parameters():
    parameter.requires_grad = False

# Set the model to evaluation mode
model.eval()

print("Feature extraction model created successfully.")



# --------------------------------------------------
# STEP 8: Create a function to extract image features
# --------------------------------------------------

def extract_features(data_loader):

    # Store extracted features and their labels
    all_features = []
    all_labels = []

    # Disable gradient calculation because the model is frozen
    with torch.no_grad():

        # Process images batch by batch
        for images, labels in data_loader:

            # Move images to GPU or CPU
            images = images.to(device)

            # Extract feature vectors from images
            features = model(images)

            # Move features to CPU and convert them to NumPy arrays
            all_features.append(features.cpu().numpy())

            # Store the original breed labels
            all_labels.append(labels.numpy())

    # Combine all batches into complete arrays
    all_features = np.concatenate(all_features)
    all_labels = np.concatenate(all_labels)

    return all_features, all_labels


print("Feature extraction function is ready.")



# --------------------------------------------------
# STEP 9: Extract training and validation features
# --------------------------------------------------

# Start measuring feature extraction time
start_time = time.time()

# Extract features from training images
print("\nExtracting training features...")
train_features, train_labels = extract_features(train_loader)

# Extract features from validation images
print("Extracting validation features...")
val_features, val_labels = extract_features(val_loader)
print("Extracting test features...")
test_features, test_labels = extract_features(test_loader)
# Calculate total feature extraction time
feature_extraction_time = time.time() - start_time

# Display feature information
print("\nFeature extraction completed!")
print("Training feature shape:", train_features.shape)
print("Validation feature shape:", val_features.shape)
print("Feature extraction time:", round(feature_extraction_time, 2), "seconds")



# --------------------------------------------------
# STEP 10: Train the Scikit-learn Linear SVM classifier
# --------------------------------------------------

# Create a Linear SVM classifier
svm_classifier = LinearSVC(
    C=1.0,
    random_state=42,
    max_iter=10000
)

# Start measuring classifier training time
start_time = time.time()

# Train the classifier using training features only
svm_classifier.fit(train_features, train_labels)

# Calculate classifier training time
classifier_training_time = time.time() - start_time

print("\nLinear SVM training completed!")
print("Classifier training time:", round(classifier_training_time, 2), "seconds")


# --------------------------------------------------
# STEP 11: Evaluate the classifier on validation data
# --------------------------------------------------

# Predict breed labels for validation images
val_predictions = svm_classifier.predict(val_features)

# Calculate validation accuracy
val_accuracy = accuracy_score(val_labels, val_predictions)

# Calculate macro F1 score
val_f1 = f1_score(
    val_labels,
    val_predictions,
    average="macro"
)
# Evaluate the SVM on the official test dataset
test_predictions = svm_classifier.predict(test_features)

test_accuracy = accuracy_score(test_labels, test_predictions)
test_f1 = f1_score(test_labels, test_predictions, average="macro")
# Create SVM confusion matrix
cm = confusion_matrix(test_labels, test_predictions)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(18, 15))

sns.heatmap(
    cm,
    cmap="Blues",
    annot=False,
    fmt="d",
    xticklabels=test_dataset.classes,
    yticklabels=test_dataset.classes
)

plt.xticks(rotation=90, fontsize=7)
plt.yticks(rotation=0, fontsize=7)

plt.title("Feature Extraction + SVM Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")

plt.tight_layout()

plt.savefig(
    "results/figures/svm_confusion_matrix.png",
    dpi=300
)

plt.close()

print("SVM confusion matrix saved successfully!")
print("\nTest Results")
print("------------------------------")
print("Test Accuracy:", round(test_accuracy, 4))
print("Test Macro F1:", round(test_f1, 4))
# Display validation results
print("\nValidation Results")
print("------------------------------")
print("Validation Accuracy:", round(val_accuracy, 4))
print("Validation Macro F1:", round(val_f1, 4))



# --------------------------------------------------
# STEP 12: Save the trained classifier and results
# --------------------------------------------------

# Create the results folders if they do not exist
os.makedirs("results/checkpoints", exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)

# Save the trained SVM classifier
with open("results/checkpoints/feature_extraction_svm.pkl", "wb") as file:
    pickle.dump(svm_classifier, file)

# Store validation results and training times
results = {
    "method": "Feature Extraction with Linear SVM",
    "validation_accuracy": float(val_accuracy),
    "validation_macro_f1": float(val_f1),
    "test_accuracy": float(test_accuracy),
    "test_macro_f1": float(test_f1),
    "feature_extraction_time": round(feature_extraction_time, 2),
    "classifier_training_time": round(classifier_training_time, 2),
    "total_time": round(
        feature_extraction_time + classifier_training_time, 2
    )
}

# Save results in a JSON file
with open("results/metrics/feature_extraction_results.json", "w") as file:
    json.dump(results, file, indent=4)

print("\nFeature extraction results saved successfully!")
print("SVM model saved successfully!")
print("Total time:", results["total_time"], "seconds")

