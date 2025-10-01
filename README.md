# Deepfake Detection Model

A deep learning project for detecting deepfake images using ResNet18 with Grad-CAM visualization.

## Project Overview

This project implements a binary classification model to distinguish between real and fake (deepfake) images. The model uses a pre-trained ResNet18 architecture with transfer learning and includes advanced visualization techniques using Grad-CAM to understand which regions of an image the model focuses on when making predictions.

## Features

- **Transfer Learning**: Uses pre-trained ResNet18 with fine-tuned final layer
- **Grad-CAM Visualization**: Generates heatmaps showing important image regions for predictions
- **Data Augmentation**: Implements color jitter and horizontal flips for robust training
- **Binary Classification**: Optimized for fake vs real image detection
- **Jupyter Notebook Interface**: Interactive development and visualization

## Project Structure

```
Model/
├── notebooks/
│   └── 03-resnet-binary.ipynb    # Main training and Grad-CAM implementation
├── data/                         # Training and validation datasets (not in git)
│   ├── train/
│   │   ├── fake/
│   │   └── real/
│   └── val/
│       ├── fake/
│       └── real/
├── dataset_raw/                  # Original dataset (not in git)
├── .gitignore
└── README.md
```

## Requirements

```bash
pip install torch torchvision
pip install opencv-python
pip install matplotlib
pip install numpy
pip install pillow
pip install scikit-learn
pip install tqdm
```

## Usage

### 1. Data Preparation
- Place your raw dataset in `dataset_raw/` directory
- Run the data splitting cells in the notebook to create train/val splits

### 2. Model Training
- Open `notebooks/03-resnet-binary.ipynb`
- Run cells sequentially to train the model
- Model uses frozen ResNet18 backbone with trainable final layer

### 3. Grad-CAM Visualization
- After training, run the Grad-CAM cells to generate heatmaps
- Visualize which regions the model considers important for predictions
- Save overlaid heatmap images for analysis

## Model Architecture

- **Base Model**: ResNet18 (pre-trained on ImageNet)
- **Modification**: Final FC layer replaced with single output for binary classification
- **Loss Function**: BCEWithLogitsLoss
- **Optimizer**: Adam (lr=1e-4)
- **Input Size**: 224x224 RGB images

## Grad-CAM Implementation

The project includes a robust Grad-CAM implementation that:
- Hooks into the last convolutional block (`model.layer4[-1]`)
- Captures feature maps and gradients during forward/backward passes
- Generates class-specific activation maps
- Overlays heatmaps on original images for visualization

## Key Features

### Data Handling
- Automatic train/validation splitting (80/20)
- Robust data loading with proper transforms
- Support for various image formats (jpg, png, jpeg)

### Training
- Transfer learning with frozen backbone
- Early stopping based on validation loss
- Comprehensive metrics tracking (loss, accuracy)

### Visualization
- Grad-CAM heatmap generation
- Proper color space handling (RGB/BGR conversions)
- High-quality overlay images

## Results

The model achieves effective binary classification on deepfake detection tasks. Grad-CAM visualizations help interpret model decisions by highlighting facial regions and artifacts that indicate fake content.

## Contributing

This is a research project for academic purposes. Feel free to fork and adapt for your own deepfake detection research.

## License

Academic use only. Please cite appropriately if used in research publications.

## Contact

For questions about this implementation, please refer to the notebook documentation or create an issue in the repository.