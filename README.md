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

### Training the model

To train the model use the trainer in `Trainer.py`
```python
data_manager = DataManager(root_dir = "./data")
model = ResNet.load_model("resnet50", n_classes = 345, in_channels = 3)

trainer = Trainer(
    model,
    data_manager,
    epochs = 10,
    batch_size = 256,
    eval_interval = 500,
    device = "cuda:0", 
    learning_rate = 0.001, 
)

trainer.train(domain = "real", run_name = "resnet50_real")
```
This requires wandb for loss and metrics logging