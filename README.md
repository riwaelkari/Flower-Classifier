# Flower Image Classification

This project is a robust image classification model designed to identify over 100 flower species using deep learning techniques. Built as part of the AWS AI & ML Nanodegree, it employs PyTorch for model development, pre-trained architectures (VGG16, ResNet18), and advanced data processing methods for high accuracy and performance.

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)

## Features
- **Advanced Data Processing**: Includes data augmentation and normalization for enhanced generalization.
- **Pre-Trained Architectures**: Utilizes VGG16 and ResNet18 for transfer learning.
- **Custom Classifier**: Designed and fine-tuned for flower classification.
- **Command-Line Applications**: Training and prediction tools with GPU acceleration, category name mapping, and top-K predictions.
- **Deployment-Ready Code**: Ensures adherence to industry standards.

## Project Structure
- `train.py`: Script for training the model.
- `predict.py`: Script for making predictions using the trained model.
- `cat_to_name.json`: JSON file mapping category IDs to flower names.
- `Image_Classifier.ipynb`: Jupyter notebook containing the code for training and evaluation.

## Setup Instructions
### Prerequisites
- Python 3.8 or higher
- PyTorch and TorchVision
- GPU (optional but recommended)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/riwaelkari/Flower-Classifier
   cd Flower-Classifier
   ``` 

2. Install required dependencies:
```bash
pip install torch torchvision numpy matplotlib json5
```

## Dataset

This project requires a flower dataset, such as the [102 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/). Download the dataset and extract it to a directory (e.g., `flowers/`). Ensure the directory structure is as follows:

```bash
flowers/
├── train/
├── valid/
└── test/
```

## Training the Model

To train the model, run:

```bash
python train.py flowers --save_dir checkpoint_dir --arch resnet18 --learning_rate 0.001 --epochs 20 --gpu
```

Replace `flowers` with the path to your dataset and `checkpoint_dir` with the directory to save the model.

## Predicting Flower Names

To predict a flower name, run:

```bash
python predict.py image_path checkpoint_dir --top_k 5 --category_names cat_to_name.json --gpu
```

Replace `image_path` with the path to the input image and `checkpoint_dir` with the trained model's checkpoint.

## Usage

### Example Prediction

```bash
python predict.py flowers/test/1/image_06743.jpg checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu
```

### Sample Output:

```plaintext
1: Pink Primrose with probability 0.87
2: Hard-Leaved Pocket Orchid with probability 0.08
3: Yellow Iris with probability 0.03
4: Giant White Arum Lily with probability 0.01
5: Fire Lily with probability 0.01
```

## Acknowledgements

- **Udacity**: AWS AI & ML Nanodegree.
- **PyTorch**: Framework used for model building.
- **Flower Dataset**: Public dataset for training and evaluation.
