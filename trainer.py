import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import math
import wandb

from utils import DataManager

class Trainer:
    def __init__(
            self, 
            model: nn.Module, 
            data_manager: DataManager, 
            epochs: int,
            batch_size: int = 64,
            eval_interval: int = 500,
            device: str = "cuda:0", 
            learning_rate: float = 0.001, 
        ) -> None:
        
        self.model = model
        self.data_manager = data_manager
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.eval_interval = eval_interval
        
        self.device = device
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3)

    def train(self, domain: str, run_name: str) -> None:

        num_samples = len(data_manager.data["train"][domain])
        num_steps = math.ceil(num_samples / self.batch_size) * self.epochs

        run = wandb.init(
            project = run_name,
            config={
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "model_type": self.model.model_name,
                "batch_size": self.self.batch_size
            }
        )

        print(f"Training for {num_steps} steps")
        
        for step in range(num_steps):
            
            self.model.train()
            
            [inputs, labels] = self.data_manager.sample(domain, split="train", batch_size=self.batch_size, return_tensors = True)
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            wandb.log({"train_loss": loss.item()})
            
            if step % self.eval_interval == 0:

                val_loss = self.evaluate(domain)
                self.scheduler.step(val_loss)

    def evaluate(self, domain: str) -> int:
        
        self.model.eval()
        
        num_samples = len(data_manager.data["test"][domain])
        eval_steps = math.ceil(num_samples / self.batch_size)
        
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():

            for step in tqdm(range(eval_steps)):
            
                inputs, labels = self.data_manager.sample(domain, split="test", batch_size=self.batch_size, return_tensors = True)
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = test_loss / num_samples
        accuracy = correct / total

        wandb.log({"test_loss": avg_loss, "accuracy": accuracy})
        
        return avg_loss