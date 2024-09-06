This repository is Work in Progress

### Preparing the Dataset

We use [DomainNet](https://ai.bu.edu/M3SDA/#overview) for all the experiments on domain adaptation. To download the dataset run
```bash
source download_dataset.sh
```

Real | Painting | Quickdraw | Clipart | Sketch
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![real](./assets/real.jpg)|![painting](./assets/painting.jpg)|![quickdraw](./assets/quickdraw.png)|![clipart](./assets/clipart.jpg)|![sketch](./assets/sketch.jpg)

This will download the dataset inside the data dir in separate train and test dirs. The bash script requires `jq`, `wget` and `unzip` to function properly, make sure to install those before running the script

It takes around 22 mins to download and extract the entire dataset which is around 15GB in size

### Model Architecture

We use ResNet Architecture based models for a starting point as they are efficient as well as the simplest models for implementation

You can create and save a resnet152 model in python using
```python
from models import ResNet

model = Resnet.load_model("resnet152", n_classes = 345)
model.save_model("resnetv2.pth")

# load the saved model
model.load_model("resnetv2.pth")
```