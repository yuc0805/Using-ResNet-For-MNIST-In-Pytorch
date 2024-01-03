# ResNet on MNIST/FashionMNIST with PyTorch

## Overview

This repository contains code to replicate the ResNet architecture on the MNIST datasets using PyTorch.

## Prerequisites

Make sure you have the following dependencies installed:

- Python 3.x
- See `requirements.txt` for specific versions of the required packages.

You can install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Repository Structure

- `main.py`: The main script to train and evaluate the ResNet model on MNIST.
- `resnet_model.py`: Implementation of the ResNet model with the ability to choose desire ResNet architecture.
- `engine_main.py`: Utility functions for data loading, training, and evaluation.
- `datasets.py`: Build MNIST with some simple data augumentation.
- `requirements.txt`: List of Python packages and versions used in this project.

## Hyperparameters

The script `main.py` is designed to be configurable through command-line arguments. Below are the key hyperparameters that can be adjusted to customize the training process:

### Training Parameters:
- `--batch_size`: Specifies the batch size for training. The number of samples that will be processed in each iteration.
- `--epochs`: Specifies the number of training epochs. An epoch is one complete pass through the entire training dataset.
- `--start_epoch`: Specifies the starting epoch for training.

### Model Parameters:
- `--input_size`: Specifies the size of the input images (assumed to be square).
- `--num_classes`: Specifies the number of classes in the classification task.
- `--model`: Specifies the name of the model architecture to train (e.g., 'resnet_18').
- `--in_channels`: Specifies the number of input channels in the images (e.g., 1 for grayscale, 3 for RGB).

### Augmentation Parameters:
- `--color_jitter`: Specifies the color jitter factor for data augmentation. This parameter controls the randomness in color transformations.
- `--random_affine`: Specifies random affine transformation arguments. It includes the maximum rotation angle, scale range, and shear range.

### Optimizer Parameters:
- `--lr`: Specifies the learning rate for the optimizer (Adam). Learning rate controls the step size during optimization.

### Dataset Parameters:
- `--output_dir`: Specifies the path where the output (e.g., model checkpoints) will be saved.
- `--device`: Specifies the device to use for training/testing (e.g., 'cpu' or 'cuda').
- `--seed`: Specifies the random seed for reproducibility.

### Other Parameters:
- `--eval`: If present, indicates that only evaluation (no training) should be performed.

Adjust these hyperparameters based on your specific requirements to tailor the training process to your needs.

## Usage

1. Clone the repository:

```bash
git clone https://github.com/your-username/resnet-mnist.git
cd resnet-mnist
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Run the main script to train and evaluate the ResNet model:

```bash
python main.py
```

## Customization

Feel free to customize the ResNet architecture by modifying the `resnet_model.py` file. You can adjust the number of layers, channels, and other hyperparameters to suit your needs.

## Citation

If you find this repository useful, please consider citing the original ResNet paper:

```bibtex
@article{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  journal={Proceedings of the IEEE conference on computer vision and pattern recognition},
  year={2016}
}
```
