# Computer Vision Homework 2
## Image Classification Using Transfer Learning

### 1. Project Overview

This project implements and compares three image classification approaches using the Oxford-IIIT Pet Dataset. The task is to classify images into 37 different cat and dog breeds.

The project uses a pretrained MobileNetV3 Small model and a traditional machine learning classifier.

### 2. Methods Implemented

Three approaches were implemented:

1. **Feature Extraction + Linear SVM:** A pretrained MobileNetV3 Small network extracts image features. A Linear Support Vector Machine performs breed classification.
2. **Classifier-Head Training:** The pretrained MobileNetV3 feature extractor is frozen, and only the classifier layers are trained.
3. **Full Fine-Tuning:** The pretrained MobileNetV3 Small network is fine-tuned, with all model parameters trainable.

### 3. Dataset

The Oxford-IIIT Pet Dataset contains 7,349 images belonging to 37 categories:

- 25 dog breeds
- 12 cat breeds

Dataset partitions:

| Partition | Images |
|---|---:|
| Training | 2,944 |
| Validation | 736 |
| Official Test | 3,669 |
| Total | 7,349 |

The official training and validation partition was divided using an 80/20 stratified split with random seed 42. The official test partition was kept separate.

Dataset source: [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)

### 4. Image Preprocessing

- Input image size: 224 × 224 pixels.
- Training augmentation: Random resized crop and random horizontal flip.
- Validation and test preprocessing: Pretrained MobileNetV3 weight transformations.
- Normalization: ImageNet mean and standard deviation.

Feature extraction uses the standard pretrained-weight transformations without random augmentation.

### 5. Model Configuration

Both MobileNetV3 training approaches use:

- Architecture: MobileNetV3 Small
- Pretrained weights: ImageNet
- Optimizer: Adam
- Loss function: CrossEntropyLoss
- Learning rate: 0.0001
- Batch size: 32
- Training epochs: 10
- Number of output classes: 37

Trainable parameters:

| Method | Trainable Parameters |
|---|---:|
| Classifier-Head Training | 628,773 |
| Full Fine-Tuning | 1,555,781 |

The Linear SVM uses `C=1.0`, `random_state=42`, and `max_iter=10000`.

### 6. Evaluation Results

All approaches were evaluated on the official test dataset.

| Method | Test Accuracy | Macro F1 |
|---|---:|---:|
| Feature Extraction + Linear SVM | 82.99% | 82.71% |
| Classifier-Head Training | **84.46%** | **84.05%** |
| Full Fine-Tuning | 83.84% | 83.51% |

Classifier-head training achieved the highest test accuracy and macro F1 score among the three approaches.

### 7. Visualizations and Analysis

The project includes:

- Training and validation accuracy graphs.
- Training and validation loss graphs.
- Confusion matrices for all three approaches.
- Correct and incorrect classification examples.
- Per-breed image counts.
- Analysis of the most frequently confused breed pairs.

The five most confused breed pairs for the classifier-head model were:

1. British Shorthair and Russian Blue: 29
2. American Pit Bull Terrier and Staffordshire Bull Terrier: 29
3. Birman and Ragdoll: 26
4. Bengal and Maine Coon: 20
5. English Cocker Spaniel and English Setter: 18

### 8. Software and Dependencies

The project was developed using:

| Software / Library | Version |
|---|---|
| Python | 3.13.17 |
| PyTorch | 2.7.1+cu118 |
| Torchvision | 0.22.1+cu118 |
| NumPy | 2.5.4 |
| Scikit-learn | 1.9.1 |
| Matplotlib | 3.11.2 |
| Pandas | 3.1.1 |
| Pillow | 13.0.0 |
| Seaborn | 0.13.2 |
| tqdm | 4.70.1 |

PyTorch was installed with CUDA 11.8 support.

### 9. Installation

Clone the repository and open its main directory.

Install the dependencies:

```bash
pip install -r requirements.txt
```

For GPU support, install a compatible PyTorch and Torchvision CUDA build.

### 10. Running the Project

Run commands from the main project directory.

**Download the dataset:**

```bash
python src/download_dataset.py
```

**Prepare the training and validation split:**

```bash
python src/prepare_splits.py
```

**Feature extraction and SVM:**

```bash
python src/extract_features.py
```

**Classifier-head training:**

```bash
python src/train_classifier.py
```

**Full fine-tuning:**

```bash
python src/train_model.py
```


**Evaluate the trained models:**

```bash
python src/evaluate_classifier_head.py
python src/evaluate_full_finetuning.py
```

**Generate plots and prediction examples:**

```bash
python src/plot_training_history.py
python src/visualize_predictions.py
python src/save_confusion_analysis.py
```

Additional scripts are available in the `src` folder for plotting training history, generating prediction examples, and analyzing confused breeds.

### 11. Reproducibility

A fixed random seed of 42 is used for data splitting and model experiments. Python, NumPy, and PyTorch random seeds are configured in the relevant scripts. Deterministic CuDNN settings are also applied.

The training and validation indices are saved in:

`data/splits/train_val_split.json`

The best neural-network checkpoints are selected using validation accuracy. The official test set is reserved for final evaluation.

### 12. Project Structure

```text
ComputerVision_HW2/
│
├── data/
│   ├── oxford_iiit_pet/
│   └── splits/
│       └── train_val_split.json
│
├── results/
│   ├── checkpoints/
│   ├── figures/
│   ├── metrics/
│   └── predictions/
│
├── src/
│   ├── download_dataset.py
│   ├── prepare_splits.py
│   ├── extract_features.py
│   ├── train_classifier.py
│   ├── train_model.py
│   ├── evaluate_classifier_head.py
│   ├── evaluate_full_finetuning.py
│   └── other analysis scripts
│
├── requirements.txt
└── README.md
```

### 13. Dataset Reference

Parkhi, O. M., Vedaldi, A., Zisserman, A., and Jawahar, C. V. (2012). Cats and Dogs. IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Dataset: Oxford-IIIT Pet Dataset.
