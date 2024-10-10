from utils import DataManager
from models import ResNet
from trainer import Trainer

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