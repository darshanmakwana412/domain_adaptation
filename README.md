
## Project Overview

This project focuses on training multiple deep learning models of the same architecture on shifted domain datasets and then developing a unified model which performs reasonably well on all the shifted domains without any training. The key components of this repository include:

- **Data Management**: Loading, preprocessing, and augmenting images from datasets such as the STL-10 dataset.
- **Model Training**: Training a ResNet model on these augmented datasets and evaluating its performance.
- **Data Augmentation**: Applying domain-specific augmentations such as adding rain, fog, and autumn effects to images.
- **Training Process**: The training process leverages PyTorch, along with advanced features like learning rate scheduling and integration with Weights and Biases (wandb) for experiment tracking.

### Directory Structure

```
.
├── app.py                     # Streamlit interface to visualise the results
├── aug                         # Data augmentation scripts
│   ├── automold_helper.py      # Helper functions for augmentations
│   ├── automold.py             # Contains functions to apply augmentations (rain, fog, autumn)
│   ├── data_manager.py         # Handles data loading and preprocessing
├── demo.ipynb                  # Jupyter notebook for demonstration
├── exp.ipynb                   # Jupyter notebook for experimental results -> includes the various approaches tried
├── fig.png                     # Image used in the notebook for visualizations -> Augmentation effect on images
├── Grad.ipynb                  # Jupyter notebook, experimenting with Grad-CAM
├── hack.py                     # Helper script with data augmentation and preprocessing
├── models.py                   # Defines the ResNet architecture and model loading logic
├── tinkering.ipynb             # Jupyter notebook for experimentation (version 1, old)
├── trainer.py                  # Trainer class that handles model training and evaluation
├── training.ipynb              # Jupyter notebook for model training
├── train.py                    # Script for training the model
└── utils.py                    # Utility functions for the project
```

## Setup and Installation

1. Clone the repository:

   ```bash
   git clone <repo_url>
   cd <repo_name>
   ```

2. Install dependencies:

   Ensure you have Python 3.7+ and `pip` installed. Then, install the required libraries by running:

   ```bash
   pip install -r requirements.txt
   ```

   The requirements include libraries such as:

   - `torch` (for PyTorch)
   - `torchvision` (for image transformations)
   - `wandb` (for experiment tracking)
   - `PIL`, `numpy`, `cv2` (for image handling and transformations)

3. Prepare the data:

   The project assumes that the STL-10 dataset (in binary format) is stored in the `./stl10_binary/` directory. Download the dataset from [here](https://cs.stanford.edu/~acoates/stl10/) and place it in the appropriate directory.

## Key Modules

### `hack.py`
This script provides functions for loading and augmenting images using techniques such as adding rain, fog, or autumn effects. It utilizes the `DataManager` class for loading the STL-10 dataset and applies transformations to images.

### `trainer.py`
The `Trainer` class is responsible for training and evaluating the deep learning model. It handles:

- Training loop: Logs training losses, optimizes the model using Adam optimizer, and applies learning rate adjustments.
- Evaluation: Evaluates the model on test data and logs metrics such as test loss and accuracy.

### `models.py`
This file contains the definition of the `ResNet` architecture and a method for loading pre-trained models. It supports various versions of ResNet (e.g., ResNet-101) and is designed to work with the custom dataset.

### `aug/automold.py`
This module contains functions that apply data augmentation techniques to images, such as adding:

- **Rain**: Simulates rainy weather conditions on images.
- **Fog**: Adds a fog effect to images.
- **Autumn**: Alters the image to simulate autumn foliage.

These augmentations are applied to training data to improve the generalization of the model.

### `utils.py`
Utility functions for various helper tasks, such as image transformations, resizing, or other preprocessing functions.

## Training the Model

To train the model, use the `train.py` script. Here's an example of how to run it:

```bash
python train.py
```

This will start the training process using the default parameters in the script. You can modify parameters such as the number of epochs, batch size, learning rate, and augmentation domain (base, rain, autumn, fog) directly in the script.

Alternatively, you can also use the `trainer.py` class for more fine-grained control over the training process:

```python
from models import ResNet
from trainer import Trainer
from aug.automold import add_rain, add_fog, add_autumn

# Initialize data manager and model
data_manager = DataManager(root_dir="./data")
model = ResNet.load_model("resnet101", n_classes=10, in_channels=3)

# Initialize and run the trainer
trainer = Trainer(
    model,
    data_manager,
    epochs=200,
    batch_size=256,
    eval_interval=200,
    device="cuda:0",
    learning_rate=0.001,
)

trainer.train(domain="base", run_name="stl10")
```

## Experiment Tracking with Weights and Biases (wandb)

The project integrates with Weights and Biases for experiment tracking. To enable this, make sure you have a wandb account and are logged in using the following command:

```bash
wandb login
```

Once logged in, the training and evaluation metrics will be automatically logged to your wandb dashboard.

## Running Notebooks

Several Jupyter notebooks are provided for experimentation and demonstration:

- **`demo.ipynb`**: Demonstrates an end-to-end usage of the classification task. Starts by loading the model, receives image input from the base as well as the shifted domains and performs the classification
- **`exp.ipynb`**: Used for running experiments and analyzing results.
- **`merging.ipynb`**: Used for merging or combining results from different models or runs.

