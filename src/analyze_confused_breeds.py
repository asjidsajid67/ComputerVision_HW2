
# --------------------------------------------------
# Computer Vision Homework 2
# Analyze Most Commonly Confused Breeds
# --------------------------------------------------

import os
import torch
import torch.nn as nn
import torchvision

from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix


# --------------------------------------------------
# STEP 1: Select device
# --------------------------------------------------

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# --------------------------------------------------
# STEP 2: Load pretrained weights
# --------------------------------------------------

weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
test_transform = weights.transforms()

print("Pretrained weights loaded.")


# --------------------------------------------------
# STEP 3: Load test dataset
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
# STEP 4: Create DataLoader
# --------------------------------------------------

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

print("Test DataLoader created successfully.")


# --------------------------------------------------
# STEP 5: Create MobileNetV3 model
# --------------------------------------------------

model = torchvision.models.mobilenet_v3_small(
    weights=weights
)

input_features = model.classifier[3].in_features

# Replace final layer with 37 classes
model.classifier[3] = nn.Linear(input_features, 37)

model = model.to(device)

print("Model created successfully.")


# --------------------------------------------------
# STEP 6: Load trained model
# --------------------------------------------------

checkpoint_path = "results/checkpoints/classifier_head_best.pth"

model.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )
)

model.eval()

print("Trained model loaded successfully!")


# --------------------------------------------------
# STEP 7: Collect predictions
# --------------------------------------------------

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = torch.argmax(outputs, dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.numpy())

print("Predictions collected successfully!")
print("Total predictions:", len(all_predictions))


# --------------------------------------------------
# STEP 8: Generate confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(37))
)

class_names = test_dataset.classes

print("Confusion matrix generated successfully.")
print("Matrix shape:", cm.shape)


# --------------------------------------------------
# STEP 9: Find most confused breed pairs
# --------------------------------------------------

confused_pairs = []

# Compare each pair of breeds
for i in range(len(class_names)):

    for j in range(i + 1, len(class_names)):

        # Count confusion in both directions
        confusion_count = int(cm[i, j] + cm[j, i])

        if confusion_count > 0:

            confused_pairs.append((
                confusion_count,
                class_names[i],
                class_names[j],
                int(cm[i, j]),
                int(cm[j, i])
            ))


# Sort from highest to lowest confusion
confused_pairs.sort(reverse=True)

print("\nTop 5 Most Confused Breed Pairs:")

for rank, pair in enumerate(confused_pairs[:5], start=1):

    count, breed_a, breed_b, a_to_b, b_to_a = pair

    print(f"\n{rank}. {breed_a} and {breed_b}")
    print("Total confusions:", count)
    print(f"{breed_a} predicted as {breed_b}:", a_to_b)
    print(f"{breed_b} predicted as {breed_a}:", b_to_a)

print("\nBreed confusion analysis completed!")
