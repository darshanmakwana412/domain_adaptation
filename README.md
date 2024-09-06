### Preparing the Dataset

We use [DomainNet](https://ai.bu.edu/M3SDA/#overview) for all the experiments on domain adaptation. To download the dataset run
```bash
source download_dataset.sh
```
This will download the dataset inside the data dir in separate train and test dirs. The script requires `jq`, `wget` and `unzip` to function properly, make sure to install those before running the script