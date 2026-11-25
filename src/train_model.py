import os
import json
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader, Subset

# --------------------------------------------------
# STEP 1: Set random seeds for reproducibility
# --------------------------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --------------------------------------------------
# STEP 2: Select the computing device
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)

# --------------------------------------------------
# STEP 3: Define dataset location
# --------------------------------------------------

data_dir = os.path.join("data", "oxford_iiit_pet")

# --------------------------------------------------
# STEP 4: Load saved training and validation indices
# --------------------------------------------------

with open("data/splits/train_val_split.json", "r") as file:
    split_data = json.load(file)

train_indices = split_data["train_indices"]
val_indices = split_data["val_indices"]

print("Training images:", len(train_indices))
print("Validation images:", len(val_indices))

# --------------------------------------------------
# STEP 5: Define image transformations
# --------------------------------------------------

# Load the preprocessing settings for MobileNetV3
weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT

# Training images: resize, crop, and apply data augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std
    )
])

# Validation images: use consistent preprocessing without augmentation
val_transform = weights.transforms()

print("Image transformations are ready.")


# --------------------------------------------------
# STEP 6: Load training and validation datasets
# --------------------------------------------------

# Load training images with data augmentation
train_dataset = OxfordIIITPet(
    root=data_dir,
    split="trainval",
    target_types="category",
    transform=train_transform,
    download=False
)

# Load validation images without data augmentation
val_dataset = OxfordIIITPet(
    root=data_dir,
    split="trainval",
    target_types="category",
    transform=val_transform,
    download=False
)

# Select the images using our saved split indices
train_dataset = Subset(train_dataset, train_indices)
val_dataset = Subset(val_dataset, val_indices)

print("Training and validation datasets loaded successfully.")
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# --------------------------------------------------
# STEP 7: Create DataLoaders
# --------------------------------------------------

# Set the batch size
batch_size = 32

# Load training and validation images in batches
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

print("DataLoaders created successfully.")
print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))

# --------------------------------------------------
# STEP 8: Test image loading
# --------------------------------------------------

images, labels = next(iter(train_loader))

print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)
print("Image loading test completed successfully.")

# --------------------------------------------------
# STEP 9: Create MobileNetV3 model
# --------------------------------------------------

def create_model():

    # Load pretrained MobileNetV3 Small
    model = torchvision.models.mobilenet_v3_small(
        weights=weights
    )

    # Get the number of input features
    input_features = model.classifier[3].in_features

    # Replace the final layer with 37 output classes
    model.classifier[3] = nn.Linear(input_features, 37)

    # Move the model to GPU or CPU
    model = model.to(device)

    return model


# Create the model
model = create_model()

print("MobileNetV3 model created successfully.")
print("Number of output classes:", model.classifier[3].out_features)

# --------------------------------------------------
# STEP 10: Count trainable parameters
# --------------------------------------------------

def count_trainable_parameters(model):

    total_parameters = 0

    for parameter in model.parameters():
        if parameter.requires_grad:
            total_parameters += parameter.numel()

    return total_parameters


trainable_parameters = count_trainable_parameters(model)

print("Trainable parameters:", trainable_parameters)

# --------------------------------------------------
# STEP 11: Define loss function and optimizer
# --------------------------------------------------

# Define the classification loss
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

# Set the number of training epochs
num_epochs = 10

print("Loss function and optimizer are ready.")
print("Learning rate:", 0.0001)
print("Training epochs:", num_epochs)

# --------------------------------------------------
# STEP 12: Define the training function
# --------------------------------------------------

def train_model(model, train_loader, criterion, optimizer):

    model.train()

    total_loss = 0
    correct_predictions = 0
    total_images = 0

    for images, labels in train_loader:

        # Move images and labels to GPU or CPU
        images = images.to(device)
        labels = labels.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Make predictions
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Calculate gradients
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Calculate training accuracy
        total_loss += loss.item()
        predictions = outputs.argmax(dim=1)

        correct_predictions += (predictions == labels).sum().item()
        total_images += labels.size(0)

    average_loss = total_loss / len(train_loader)
    accuracy = correct_predictions / total_images

    return average_loss, accuracy


print("Training function is ready.")

# --------------------------------------------------
# STEP 13: Define the validation function
# --------------------------------------------------

def validate_model(model, val_loader, criterion):

    model.eval()

    total_loss = 0
    correct_predictions = 0
    total_images = 0

    # Disable gradient calculation during validation
    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Make predictions
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            # Calculate accuracy
            predictions = outputs.argmax(dim=1)

            correct_predictions += (predictions == labels).sum().item()
            total_images += labels.size(0)

    average_loss = total_loss / len(val_loader)
    accuracy = correct_predictions / total_images

    return average_loss, accuracy


print("Validation function is ready.")

# --------------------------------------------------
# STEP 14: Train the model
# --------------------------------------------------

# Store training results
history = []

# Track the best validation accuracy
best_accuracy = 0

# Start training timer
start_time = time.time()

for epoch in range(num_epochs):

    print(f"\nEpoch [{epoch + 1}/{num_epochs}]")

    # Train the model
    train_loss, train_accuracy = train_model(
        model, train_loader, criterion, optimizer
    )

    # Validate the model
    val_loss, val_accuracy = validate_model(
        model, val_loader, criterion
    )

    # Save results for this epoch
    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy
    })

    print(f"Training Loss: {train_loss:.4f}")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    # Save the best model
    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "results/checkpoints/full_finetuning_best.pth"
        )

        print("Best model saved!")

# Calculate total training time
training_time = time.time() - start_time

# Save training history and training time
with open("results/metrics/full_finetuning_history.json", "w") as file:
    json.dump({
        "history": history,
        "training_time_seconds": round(training_time, 2)
    }, file, indent=4)

print("\nFull fine-tuning completed!")
print("Best validation accuracy:", round(best_accuracy, 4))
print("Training time:", round(training_time, 2), "seconds")