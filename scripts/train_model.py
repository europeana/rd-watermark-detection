from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os
import pytorch_lightning as pl
import numpy as np

class ImageDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=4, val_split=0.2, test_split=0.1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def setup(self, stage=None):
        # Load the entire dataset
        full_dataset = datasets.ImageFolder(self.data_dir, transform=self.transform)

        # Splitting the dataset into train, validation, and test sets
        train_val_split = int(len(full_dataset) * (1 - self.test_split))
        train_split = int(train_val_split * (1 - self.val_split))
        
        indices = np.arange(len(full_dataset))
        np.random.shuffle(indices)
        
        train_indices = indices[:train_split]
        val_indices = indices[train_split:train_val_split]
        test_indices = indices[train_val_split:]
        
        self.train_dataset = Subset(full_dataset, train_indices)
        self.val_dataset = Subset(full_dataset, val_indices)
        self.test_dataset = Subset(full_dataset, test_indices)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


import torch
from torchvision import models
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import Accuracy

class ResNetModel(LightningModule):
    def __init__(self, num_classes, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        # Load a pre-trained ResNet-18 model
        self.model = models.resnet18(pretrained=True)
        # Replace the classifier layer for fine-tuning
        num_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_features, num_classes)
        # Metrics
        # Initialize Accuracy with the appropriate task
        if num_classes == 2:
            self.train_acc = Accuracy(num_classes=num_classes, average='macro', task='binary')
            self.val_acc = Accuracy(num_classes=num_classes, average='macro', task='binary')
            self.test_acc = Accuracy(num_classes=num_classes, average='macro', task='binary')
        else:
            self.train_acc = Accuracy(num_classes=num_classes, average='macro', task='multiclass')
            self.val_acc = Accuracy(num_classes=num_classes, average='macro', task='multiclass')
            self.test_acc = Accuracy(num_classes=num_classes, average='macro',task='multiclass')


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        # Log training accuracy
        self.log('train_loss', loss)
        self.log('train_acc', self.train_acc(outputs, labels), on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        # Log validation accuracy
        self.log('val_loss', loss)
        self.log('val_acc', self.val_acc(outputs, labels), on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        # Log test accuracy
        self.log('test_loss', loss)
        self.log('test_acc', self.test_acc(outputs, labels))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer




from pytorch_lightning import loggers as pl_loggers

# Set up TensorBoard Logger
#tb_logger = pl_loggers.TensorBoardLogger('/storage/logs/')

# You can adjust the logging level in PyTorch Lightning as follows:
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.INFO) 

# Set up data
print("Setting up data...")
data_module = ImageDataModule(data_dir='/storage/data/labeled_55255', batch_size=32)
print("Setting up model...")
# Initialize model
model = ResNetModel(num_classes=2)  # Set the number of classes based on your dataset

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ProgressBar

from pytorch_lightning.callbacks import Callback

class VerboseCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % 50 == 0:  # Log every 50 batches
            loss = outputs['loss'].item()
            print(f"Batch {batch_idx}: Loss {loss}")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx % 50 == 0:
            val_loss = outputs['loss'].item()
            print(f"Validation Batch {batch_idx}: Loss {val_loss}")


print("Training...")
# Initialize a trainer with a logger and more detailed progress bar settings
trainer = Trainer(
    max_epochs=10,
    devices=1 if torch.cuda.is_available() else None,
    #accelerator="gpu" if torch.cuda.is_available() else None,
    #logger=tb_logger,  # Use the TensorBoard logger
    callbacks=[ProgressBar(),VerboseCallback()],  # Progress bar refreshes every batch
    enable_checkpointing=True,  # Ensures that model checkpoints are saved
    enable_progress_bar=True  # Enables the default progress bar with detailed information
)

# Train the model
trainer.fit(model, datamodule=data_module)
trainer.test(model, datamodule=data_module)