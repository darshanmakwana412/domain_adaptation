import os
import random
from PIL import Image
import numpy as np
from typing import Optional, Union
from aug.automold import add_rain, add_fog, add_autumn

import torch
import cv2
from torchvision import transforms

HEIGHT = 96
WIDTH = 96
DEPTH = 3

SIZE = HEIGHT * WIDTH * DEPTH

class DataManager:
    def __init__(self, root_dir: str = "./data", size: tuple[int, int] = (256, 256)) -> None:

        self.size = size
        self.root_dir = root_dir
        self.num_classes = 10
        
        self.domains = ["base", "rain", "autumn", "fog"]


        self.transforms = transforms.ToTensor()

        num_class = {}
        self.data = {
            "train": {},
            "test": {}
        }

        path_to_images = "./stl10_binary/train_X.bin"
        path_to_labels = "./stl10_binary/train_y.bin"

        with open(path_to_images, 'rb') as f:
            images = np.fromfile(f, dtype=np.uint8)
            images = np.reshape(images, (-1, 3, 96, 96))
            self.data["train"]["images"] = np.transpose(images, (0, 3, 2, 1))

        with open(path_to_labels, 'rb') as f:
            self.data["train"]["labels"] = np.fromfile(f, dtype=np.uint8) - 1

        path_to_images = "./stl10_binary/test_X.bin"
        path_to_labels = "./stl10_binary/test_y.bin"

        with open(path_to_images, 'rb') as f:
            images = np.fromfile(f, dtype=np.uint8)
            images = np.reshape(images, (-1, 3, 96, 96))
            self.data["test"]["images"] = np.transpose(images, (0, 3, 2, 1))

        with open(path_to_labels, 'rb') as f:
            labels = np.fromfile(f, dtype=np.uint8)
            self.data["test"]["labels"] = labels - 1

    def sample(self, domain: str = "base", split: str = "train", batch_size: int = 1, return_tensors: bool = False) -> Union[
        Optional[tuple[Image.Image, int]],
        Optional[tuple[list[Image.Image], list[int]]],
        Optional[tuple[torch.Tensor, torch.Tensor]]
    ]:
        new_size = (256, 256)

        samples = np.random.randint(0, 5000, batch_size)
        
        images = self.data[split]["images"][samples]
        labels = self.data[split]["labels"][samples]
        
        if domain == "rain":
            images = [add_rain(cv2.resize(img, new_size)) for img in images]
#             images = [add_rain(img) for img in images]
        elif domain == "fog":
            images = [add_fog(cv2.resize(img, new_size)) for img in images]
#             images = [add_fog(img) for img in images]
        elif domain == "autumn":
            images = [add_autumn(cv2.resize(img, new_size)) for img in images]
        elif domain == "base":
            images = [cv2.resize(img, new_size) for img in images]

        if return_tensors:
            images = torch.stack([self.transforms(img) for img in images])
            labels = torch.tensor(labels)
            return (images, labels)
        else:
            if batch_size == 1:
                return (images[0], labels[0])
            else:
                return (images, labels)

from models import ResNet
from trainer import Trainer

data_manager = DataManager(root_dir = "./data")
model = ResNet.load_model("resnet101", n_classes = 10, in_channels = 3)

trainer = Trainer(
    model,
    data_manager,
    epochs = 200,
    batch_size = 256,
    eval_interval = 200,
    device = "cuda:0", 
    learning_rate = 0.001, 
)

trainer.train(domain = "base", run_name = "stl10")