
# --------------------------------------------------
# Computer Vision Homework 2
# Visualize Correct and Incorrect Predictions
# --------------------------------------------------

import os
import torch
import torch.nn as nn
import torchvision

from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# --------------------------------------------------
# STEP 1: Select device
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# --------------------------------------------------
# STEP 2: Load pretrained MobileNetV3 weights
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
    torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )
)

model.eval()

print("Best classifier-head model loaded successfully!")

# --------------------------------------------------
# STEP 7: Collect model predictions
# --------------------------------------------------

all_predictions = []
all_labels = []

model.eval()

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = torch.argmax(outputs, dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.numpy())

print("Predictions collected successfully!")
print("Total predictions:", len(all_predictions))
print("Total labels:", len(all_labels))


# --------------------------------------------------
# STEP 8: Identify correct and incorrect predictions
# --------------------------------------------------

correct_indices = []
incorrect_indices = []

for i in range(len(all_labels)):

    if all_predictions[i] == all_labels[i]:
        correct_indices.append(i)

    else:
        incorrect_indices.append(i)

print("Correct predictions:", len(correct_indices))
print("Incorrect predictions:", len(incorrect_indices))

# --------------------------------------------------
# STEP 9: Get breed names
# --------------------------------------------------

class_names = test_dataset.classes

print("Breed names loaded successfully!")
print("Total breed classes:", len(class_names))

print("\nFirst 10 breed names:")

for name in class_names[:10]:
    print(name)

# --------------------------------------------------
# STEP 10: Create prediction visualization folder
# --------------------------------------------------

output_dir = "results/predictions"

os.makedirs(output_dir, exist_ok=True)

print("Prediction folder is ready:", output_dir)

# --------------------------------------------------

# --------------------------------------------------
# STEP 11: Select diverse correct prediction examples
# --------------------------------------------------

correct_examples = []
seen_correct_breeds = set()

for index in correct_indices:

    actual = class_names[all_labels[index]]

    if actual not in seen_correct_breeds:

        correct_examples.append(index)
        seen_correct_breeds.add(actual)

    if len(correct_examples) == 5:
        break

print("\nSelected diverse correct predictions:")

for index in correct_examples:

    actual = class_names[all_labels[index]]
    predicted = class_names[all_predictions[index]]

    print("Image:", index)
    print("Actual breed:", actual)
    print("Predicted breed:", predicted)
    print("-----------------------------")

# --------------------------------------------------

# --------------------------------------------------
# STEP 12: Select diverse incorrect prediction examples
# --------------------------------------------------

incorrect_examples = []
seen_incorrect_breeds = set()

for index in incorrect_indices:

    actual = class_names[all_labels[index]]

    if actual not in seen_incorrect_breeds:

        incorrect_examples.append(index)
        seen_incorrect_breeds.add(actual)

    if len(incorrect_examples) == 5:
        break

print("\nSelected diverse incorrect predictions:")

for index in incorrect_examples:

    actual = class_names[all_labels[index]]
    predicted = class_names[all_predictions[index]]

    print("Image:", index)
    print("Actual breed:", actual)
    print("Predicted breed:", predicted)
    print("-----------------------------")


# --------------------------------------------------
# STEP 13: Create prediction visualization function
# --------------------------------------------------

def visualize_examples(indices, title, filename):

    plt.figure(figsize=(15, 6))

    for position, index in enumerate(indices):

        image, label = test_dataset[index]

        
        # Reverse ImageNet normalization for correct display
        mean = torch.tensor(test_transform.mean).view(3, 1, 1)
        std = torch.tensor(test_transform.std).view(3, 1, 1)

        image = image * std + mean
        image = image.clamp(0, 1)

        # Convert image tensor to display format
        image = image.permute(1, 2, 0).numpy()


        actual = class_names[all_labels[index]]
        predicted = class_names[all_predictions[index]]

        plt.subplot(1, 5, position + 1)

        plt.imshow(image)
        plt.title(
            f"Actual: {actual}\nPredicted: {predicted}",
            fontsize=8
        )

        plt.axis("off")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(output_dir, filename)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()

    print("Figure saved:", save_path)

# --------------------------------------------------
# STEP 14: Generate prediction visualizations
# --------------------------------------------------

visualize_examples(
    correct_examples,
    "Correctly Classified Predictions",
    "correct_predictions.png"
)

visualize_examples(
    incorrect_examples,
    "Incorrectly Classified Predictions",
    "incorrect_predictions.png"
)

print("\nPrediction visualization completed successfully!")
