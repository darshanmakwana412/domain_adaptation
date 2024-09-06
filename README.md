This is Repository is Work in Progress

### Preparing the Dataset

We use [DomainNet](https://ai.bu.edu/M3SDA/#overview) for all the experiments on domain adaptation. To download the dataset run
```bash
source download_dataset.sh
```

Real | Painting | Quickdraw | Clipart | Sketch
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![real](./assets/real.jpg)|![painting](./assets/painting.jpg)|![quickdraw](./assets/quickdraw.png)|![clipart](./assets/clipart.jpg)|![sketch](./assets/sketch.jpg)

This will download the dataset inside the data dir in separate train and test dirs. The script requires `jq`, `wget` and `unzip` to function properly, make sure to install those before running the script

It takes around 22 mins to download and extract the entire dataset which is around 15GB in size