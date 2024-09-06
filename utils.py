import os
import random
from PIL import Image
from typing import Optional, Union

import torch
from torchvision import transforms

class DataManager:
    def __init__(self, root_dir: str = "./data", size: tuple[int, int] = (256, 256)) -> None:
        
        self.domains = [dirs for dirs in os.listdir(root_dir) if dirs not in ["train", "test"]]
        print(f"{len(self.domains)} domains found: {self.domains}")

        self.size = size
        self.root_dir = root_dir

        self.transforms = transforms.ToTensor()

        num_class = {}
        self.data = {
            "train": {},
            "test": {}
        }
        for domain in self.domains:
            domain_dir = os.path.join(root_dir, domain)
            for class_dir in os.listdir(domain_dir):
                if class_dir not in num_class:
                    num_class[class_dir] = 0
                num_class[class_dir] += 1

            for split in ["train", "test"]:
                with open(os.path.join(root_dir, split, f"{domain}_{split}.txt")) as f:
                    self.data[split][domain] = [[line.strip().split()[0], int(line.strip().split()[1])] for line in f.readlines()]

        underact = 0
        for key, val in num_class.items():
            if val != len(self.domains):
                print(f"class {key} only found in {val} domains")
                underact += 1

        if underact:
            print(f"{underact} classes are in minority across the domains")
        else:
            print(f"All classes are present across all the domains")

        print(f"Total number of classes: {len(num_class)}")

    def sample(self, domain: str, split: str = "train", batch_size: int = 1, return_tensors: bool = False) -> Union[
        Optional[tuple[Image.Image, int]],
        Optional[tuple[list[Image.Image], list[int]]],
        Optional[tuple[torch.Tensor, torch.Tensor]]
    ]:

        """
        Samples data points from the specified domain and split
    
        Args:
            domain (str): The domain to sample from. Must be one of the available domains in the dataset.
            split (str, optional): The dataset split to use. Defaults to "train".
            batch_size (int, optional): Number of samples to return. Defaults to 1.
            return_tensors (bool, optional): If True, returns PyTorch tensors instead of PIL images. Defaults to False.
    
        Returns:
            Union[
                Optional[tuple[Image.Image, int]],
                Optional[tuple[list[Image.Image], list[int]]],
                Optional[tuple[torch.Tensor, torch.Tensor]]
            ]
        """
        
        if domain not in self.domains:
            print(f"Domain: {domain} not found in the dataset, available domains are: {self.domains}")
            return None

        samples = random.choices(self.data[split][domain], k=batch_size)
        
        images = []
        labels = []

        for img_path, idx in samples:
            img = Image.open(os.path.join(self.root_dir, img_path)).convert("RGB").resize(self.size)
            images.append(img)
            labels.append(idx)

        if return_tensors:
            images = torch.stack([self.transforms(img) for img in images])
            labels = torch.tensor(labels)
            return (images, labels)
        else:
            if batch_size == 1:
                return (images[0], labels[0])
            else:
                return (images, labels)