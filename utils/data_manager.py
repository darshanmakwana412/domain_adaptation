import os

import numpy as np
from torchvision import datasets
from PIL import Image
import cv2

from utils import (
    automold as am,
    automold_helper as hp
)

class DataManager:

    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir + "/base_model/"
        self.train_dir = os.path.join(self.data_dir, "train")
        self.test_dir = os.path.join(self.data_dir, "test")
    
    def _download_data(self) -> None:
        cifar10_train = datasets.CIFAR10(root=self.train_dir, train=True, download=True)
        cifar10_test = datasets.CIFAR10(root=self.test_dir, train=False, download=True)
        print("Data downloaded successfully")
    
    def _process_data(self) -> None:
        '''
        In order to use the augmentation functions, it is necessary to store the images in a .jpg format.

        This function converts the bathces-py images to the following structure:
        - base_model/
            - train/
                - 0/
                    - img1.jpg
                    - img2.jpg
                    - ...
                - 1/
                    - img1.jpg
                    - img2.jpg

            - test/
                - 0/    
                    - img1.jpg  
                    - img2.jpg
                    - ...
                - 1/
                    - img1.jpg  
        '''


        train_data = datasets.CIFAR10(root=self.train_dir, train=True, download=False)
        for i in range(10):
            os.makedirs(os.path.join(self.train_dir, str(i)), exist_ok=True)

        for i, (img, label) in enumerate(train_data):
            img = Image.fromarray(img)
            img.save(os.path.join(self.train_dir, str(label), f"{i}.jpg"))

        print("Train data processed successfully")

        test_data = datasets.CIFAR10(root=self.test_dir, train=False, download=False)
        for i in range(10):
            os.makedirs(os.path.join(self.test_dir, str(i)), exist_ok=True)

        for i, (img, label) in enumerate(test_data):
            img = Image.fromarray(img)
            img.save(os.path.join(self.test_dir, str(label), f"{i}.jpg"))
        
        print("Test data processed successfully")

    def _augment_data(self, augment_type: str) -> None:
        '''
        This function applies the augmentation functions to the augmentations:
         - Rain
         - Fog
         - Autumn
        
        The augmented images are stored in the following structure:
        - rain/
            - train/
                - 0/
                    - img1.jpg
                    - img2.jpg
                    - ...
                - 1/
                    - img1.jpg
                    - img2.jpg
                    - ...
                - ...
        '''
        dir_path = os.path.join(self.data_dir, augment_type)
        os.makedirs(dir_path, exist_ok=True)
        os.makedirs(os.path.join(dir_path, "train"), exist_ok=True)
        os.makedirs(os.path.join(dir_path, "test"), exist_ok=True)

        for i in range(10):
            os.makedirs(os.path.join(dir_path, "train", str(i)), exist_ok=True)
            os.makedirs(os.path.join(dir_path, "test", str(i)), exist_ok=True)

        for i in range(10):
            train_path = os.path.join(self.train_dir, str(i))
            test_path = os.path.join(self.test_dir, str(i))

            train_path_fmt = train_path + "/*.jpg"
            test_path_fmt = test_path + "/*.jpg"

            print(f"Train path: {train_path_fmt}")
            print(f"Test path: {test_path_fmt}")

            if augment_type == "rain":
                images_train = hp.load_images(train_path_fmt)
                images_test = hp.load_images(test_path_fmt)
                rain_train = am.add_rain(images_train)
                rain_test = am.add_rain(images_test)

                for i in range(len(rain_train)):
                    rain_train_path = os.path.join(dir_path, "train", str(i))
                    cv2.imwrite(f"{rain_train_path}/{i}.jpg", rain_train[i])
                
                for i in range(len(rain_test)):
                    rain_test_path = os.path.join(dir_path, "test", str(i))
                    cv2.imwrite(f"{rain_test_path}/{i}.jpg", rain_test[i])
                
                print("Rain added successfully")

            elif augment_type == "fog":
                images_train = hp.load_images(train_path_fmt)
                images_test = hp.load_images(test_path_fmt)
                fog_train = am.add_fog(images_train)
                fog_test = am.add_fog(images_test)

                for i in range(len(fog_train)):
                    fog_train_path = os.path.join(dir_path, "train", str(i))
                    cv2.imwrite(f"{fog_train_path}/{i}.jpg", fog_train[i])
                
                
                for i in range(len(fog_test)):
                    fog_test_path = os.path.join(dir_path, "test", str(i))
                    cv2.imwrite(f"{fog_test_path}/{i}.jpg", fog_test[i])

                print("Fog added successfully")

            elif augment_type == "autumn":
                images_train = hp.load_images(train_path_fmt)
                images_test = hp.load_images(test_path_fmt)
                autumn_train = am.add_autumn(images_train)
                autumn_test = am.add_autumn(images_test)

                for i in range(len(autumn_train)):
                    autumn_train_path = os.path.join(dir_path, "train", str(i))
                    cv2.imwrite(f"{autumn_train_path}/{i}.jpg", autumn_train[i])
                
                for i in range(len(autumn_test)):
                    autumn_test_path = os.path.join(dir_path, "test", str(i))
                    cv2.imwrite(f"{autumn_test_path}/{i}.jpg", autumn_test[i])

                print("Autumn added successfully")

    def prepare_data(self, augment_type: str) -> None:
        self._download_data()
        self._process_data()
        self._augment_data(augment_type)
    
def main():
    DATA_DIR = "data"
    augment_types = ["rain", "fog", "autumn"]
    data_manager = DataManager(DATA_DIR)
    for augment_type in augment_types:
        data_manager.prepare_data(augment_type)

if __name__ == "__main__":
    main()





        
        