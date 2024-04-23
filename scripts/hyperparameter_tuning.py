import os
import fire
import torch
import tempfile
import pytorch_lightning as pl
import torch.nn.functional as F
from filelock import FileLock
from torchmetrics import Accuracy
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt


from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer

import numpy as np
from pathlib import Path
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from PIL import Image
import torchvision
import torch.nn as nn
from torchvision.models import ResNet18_Weights

class Classifier(pl.LightningModule):
    def __init__(self, config):
        super(Classifier, self).__init__()
        self.accuracy = Accuracy(task="binary", num_classes=2, top_k=1)

        output_dim = 2

        self.model = torchvision.models.resnet18(weights = ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, output_dim)
        self.sm = nn.Softmax(dim=1)

        self.lr = config["lr"]

        self.eval_loss = []
        self.eval_accuracy = []

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def forward(self, x):
        x = self.model(x)
        x = self.sm(x)
        return x

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        accuracy = self.accuracy(logits, y)
        self.log("ptl/train_loss", loss)
        self.log("ptl/train_accuracy", accuracy)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        accuracy = self.accuracy(logits, y)
        self.eval_loss.append(loss)
        self.eval_accuracy.append(accuracy)
        return {"val_loss": loss, "val_accuracy": accuracy}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.eval_loss).mean()
        avg_acc = torch.stack(self.eval_accuracy).mean()
        self.log("ptl/val_loss", avg_loss, sync_dist=True)
        self.log("ptl/val_accuracy", avg_acc, sync_dist=True)
        self.eval_loss.clear()
        self.eval_accuracy.clear()
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer



class TrainingDataset(Dataset):
    def __init__(self, imgs,labels, transform=None):
        self.transform = transform
        self.img_list = imgs
        self.labels = labels
        #self.encoding_dict = le.classes_

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.img_list[idx]
        label = self.labels[idx]

        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image,label


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=128,data_dir=None,num_workers = 6,sample = 1.0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample = sample

        self.train_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([transforms.Grayscale(num_output_channels=3)], p=0.25),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        self.test_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


    def setup(self, stage=None):
        data_dir = Path(self.data_dir)
        imgs = np.array([str(p) for p in data_dir.rglob("*/*")])
        labels = np.array([Path(p).parent.name for p in imgs])

        if self.sample != 1.0:
            _, imgs, _, labels = train_test_split(imgs, labels, stratify=labels,
                                                  test_size=self.sample, random_state=42)

        
        le = preprocessing.LabelEncoder()
        _labels = le.fit_transform(labels)
        classes = le.classes_
        labels = F.one_hot(torch.from_numpy(_labels)).float()

        X_train_val, X_test, y_train_val, y_test = train_test_split(imgs, labels, test_size=0.2,stratify=labels)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2,stratify=y_train_val)

        self.train_dataset = TrainingDataset(X_train,y_train,transform = self.train_transforms)
        self.val_dataset = TrainingDataset(X_val,y_val,transform = self.test_transforms)
        self.test_dataset = TrainingDataset(X_test,y_test,transform = self.test_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


def main(*args,**kwargs):

    data_dir =  kwargs.get('data_dir')
    saving_dir = kwargs.get('saving_dir')
    sample = kwargs.get('sample',1.0)
    num_epochs = kwargs.get('num_epochs',10)
    num_samples = kwargs.get('num_samples',30)
    grace_period = kwargs.get('grace_period',5)

    ray.init(_temp_dir=saving_dir)

    saving_dir = Path(saving_dir)
    saving_dir.mkdir(exist_ok = True, parents = True)

    default_config = {
        "lr": 1e-5,
    }

    search_space = {
        "lr": tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice([8,16,32,64,128]),
    }

    def train_func(config):
        dm = DataModule(batch_size=config["batch_size"],data_dir = data_dir, sample = sample)
        model = Classifier(config)

        trainer = pl.Trainer(
            devices="auto",
            accelerator="auto",
            strategy=RayDDPStrategy(),
            callbacks=[RayTrainReportCallback()],
            plugins=[RayLightningEnvironment()],
            enable_progress_bar=False,
            max_epochs=num_epochs
        )
        trainer = prepare_trainer(trainer)
        trainer.fit(model, datamodule=dm)


    #scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    scaling_config = ScalingConfig(
        num_workers=1, use_gpu=True, resources_per_worker={"CPU": 1, "GPU": 1}
    )

    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="ptl/val_accuracy",
            checkpoint_score_order="max",
        ),
    )

    # Define a TorchTrainer without hyper-parameters for Tuner
    ray_trainer = TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    def tune_mnist_asha(num_samples=10):
        scheduler = ASHAScheduler(max_t=num_epochs, grace_period=grace_period, reduction_factor=2)

        tuner = tune.Tuner(
            ray_trainer,
            param_space={"train_loop_config": search_space},
            tune_config=tune.TuneConfig(
                metric="ptl/val_accuracy",
                mode="max",
                num_samples=num_samples,
                scheduler=scheduler,
            ),
        )
        return tuner.fit()


    results = tune_mnist_asha(num_samples=num_samples)
    df = results.get_dataframe()
    df.to_csv(saving_dir.joinpath('hyperparameter_results.csv'))

    x = df['config/train_loop_config/batch_size'].values
    y = df['config/train_loop_config/lr'].values

    fig, ax = plt.subplots()
    z = df['ptl/val_loss'].values
    sc = ax.scatter(x, y, c=z, alpha=0.5)
    fig.colorbar(sc)
    ax.set_title('Loss')
    ax.set_yscale('log')
    fig.savefig(saving_dir.joinpath('loss_scatter.jpg'))

    fig, ax = plt.subplots()
    z = df['ptl/val_accuracy'].values
    sc = plt.scatter(x, y, c=z, alpha=0.5)
    fig.colorbar(sc)
    ax.set_title('Accuracy')
    ax.set_yscale('log')
    fig.savefig(saving_dir.joinpath('accuracy_scatter.jpg'))
    print('Finished')

      
if __name__ == "__main__":
    fire.Fire(main)