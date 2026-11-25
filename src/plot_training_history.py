import os
import json
import matplotlib.pyplot as plt
# --------------------------------------------------
# STEP 1: Load training histories
# --------------------------------------------------

with open("results/metrics/classifier_head_history.json", "r") as file:
    classifier_history = json.load(file)

with open("results/metrics/full_finetuning_history.json", "r") as file:
    finetuning_history = json.load(file)

print("Training histories loaded successfully!")

# --------------------------------------------------
# STEP 2: Prepare epoch numbers
# --------------------------------------------------

epochs = [item["epoch"] for item in classifier_history]

print("Epochs:", epochs)
print("Total epochs:", len(epochs))

# --------------------------------------------------
# STEP 3: Create output folder
# --------------------------------------------------

os.makedirs("results/figures", exist_ok=True)

print("Figures folder is ready.")
# --------------------------------------------------
# STEP 4: Classifier-head accuracy graph
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    [item["train_accuracy"] * 100 for item in classifier_history],
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    epochs,
    [item["val_accuracy"] * 100 for item in classifier_history],
    marker="o",
    label="Validation Accuracy"
)

plt.title("Classifier-Head Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.xticks(epochs)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "results/figures/classifier_head_accuracy.png",
    dpi=300
)

plt.close()

print("Classifier-head accuracy graph saved!")

# --------------------------------------------------
# STEP 5: Classifier-head loss graph
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    [item["train_loss"] for item in classifier_history],
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs,
    [item["val_loss"] for item in classifier_history],
    marker="o",
    label="Validation Loss"
)

plt.title("Classifier-Head Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(epochs)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "results/figures/classifier_head_loss.png",
    dpi=300
)

plt.close()

print("Classifier-head loss graph saved!")

# --------------------------------------------------
# STEP 6: Full fine-tuning accuracy graph
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    [item["train_accuracy"] * 100 for item in finetuning_history],
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    epochs,
    [item["val_accuracy"] * 100 for item in finetuning_history],
    marker="o",
    label="Validation Accuracy"
)

plt.title("Full Fine-Tuning Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.xticks(epochs)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "results/figures/full_finetuning_accuracy.png",
    dpi=300
)

plt.close()

print("Full fine-tuning accuracy graph saved!")
# --------------------------------------------------
# STEP 7: Full fine-tuning loss graph
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    [item["train_loss"] for item in finetuning_history],
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs,
    [item["val_loss"] for item in finetuning_history],
    marker="o",
    label="Validation Loss"
)

plt.title("Full Fine-Tuning Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(epochs)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "results/figures/full_finetuning_loss.png",
    dpi=300
)

plt.close()

print("Full fine-tuning loss graph saved!")