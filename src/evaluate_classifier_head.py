import os
import json
import torch
import torch.nn as nn
import torchvision

from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# STEP 1: Select device
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)

# --------------------------------------------------
# STEP 2: Load MobileNetV3 weights
# --------------------------------------------------

weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT

test_transform = weights.transforms()

print("Pretrained MobileNetV3 weights loaded.")

# --------------------------------------------------
# STEP 3: Load official test dataset
# --------------------------------------------------

data_dir = os.path.join("data", "oxford_iiit_pet")

test_dataset = OxfordIIITPet(
    root=data_dir,
    split="test",
    target_types="category",
    transform=test_transform,
    download=False
)

print("Test dataset loaded successfully.")
print("Test images:", len(test_dataset))

# --------------------------------------------------
# STEP 4: Create test DataLoader
# --------------------------------------------------

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

print("Test DataLoader created successfully.")
print("Test batches:", len(test_loader))

# --------------------------------------------------
# STEP 5: Recreate MobileNetV3 classifier-head model
# --------------------------------------------------

model = torchvision.models.mobilenet_v3_small(
    weights=weights
)

# Replace final classification layer
input_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(input_features, 37)

# Move model to device
model = model.to(device)

print("Classifier-head model architecture created.")

# --------------------------------------------------
# STEP 6: Load the best trained model
# --------------------------------------------------

checkpoint_path = "results/checkpoints/classifier_head_best.pth"

model.load_state_dict(
    torch.load(checkpoint_path, map_location=device, weights_only=True)
)

model.eval()

print("Best classifier-head model loaded successfully!")

# --------------------------------------------------
# STEP 7: Evaluate model on test dataset
# --------------------------------------------------

all_predictions = []
all_labels = []

model.eval()

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.numpy())

print("Test predictions completed!")
print("Total test predictions:", len(all_predictions))

# --------------------------------------------------
# STEP 8: Calculate test metrics
# --------------------------------------------------

test_accuracy = accuracy_score(all_labels, all_predictions)

test_f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro"
)

print("\nClassifier-Head Test Results")
print("--------------------------------")
print("Test Accuracy:", round(test_accuracy, 4))
print("Test Macro F1:", round(test_f1, 4))
# --------------------------------------------------
# STEP 9: Generate confusion matrix
# --------------------------------------------------

cm = confusion_matrix(all_labels, all_predictions)

os.makedirs("results/figures", exist_ok=True)

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

plt.title("Classifier-Head Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")

plt.tight_layout()

plt.savefig(
    "results/figures/classifier_head_confusion_matrix.png",
    dpi=300
)

plt.close()

print("Confusion matrix saved successfully!")
# --------------------------------------------------
# STEP 10: Save test results
# --------------------------------------------------

results = {
    "method": "Classifier-Head Training",
    "test_accuracy": float(test_accuracy),
    "test_macro_f1": float(test_f1)
}

os.makedirs("results/metrics", exist_ok=True)

with open(
    "results/metrics/classifier_head_test_results.json",
    "w"
) as file:
    json.dump(results, file, indent=4)

print("Test results saved successfully!")