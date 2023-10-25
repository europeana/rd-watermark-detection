
import fire
from pathlib import Path
import torchvision
from torchvision import transforms
import numpy as np
from sklearn import preprocessing
import torch
import torchmetrics
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
import json

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from torch.utils.data import Dataset, DataLoader
from PIL import Image

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

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




class Classifier(pl.LightningModule): 
  
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        output_dim = kwargs.get('output_dim')
        learning_rate = kwargs.get('learning_rate')
        self.threshold = kwargs.get('threshold')

        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, output_dim)
        self.sm = nn.Softmax(dim=1)
        self.accuracy = torchmetrics.Accuracy(task='binary')

    def forward(self, x):
        out = self.model(x)
        out = self.sm(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        y_hat = torch.where(y_hat>self.threshold,1,0).int()
        self.accuracy(y_hat, y.int())
        self.log('valid_acc_step', self.accuracy)
        self.log('valid_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        loss = F.cross_entropy(y_hat, y)
        y_hat = torch.where(y_hat>self.threshold,1,0).int()
        self.accuracy(y_hat, y.int())
        self.log('test_acc_step', self.accuracy)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def main(**kwargs):

    data_dir = kwargs.get('data_dir')
    saving_dir = kwargs.get('saving_dir')
    max_epochs = kwargs.get('max_epochs',10)
    sample = kwargs.get('sample',1.0)
    test_size = kwargs.get('test_size',0.2)
    batch_size = kwargs.get('batch_size',16)
    learning_rate = kwargs.get('learning_rate',1e-4)
    threshold = kwargs.get('threshold',0.5)
    num_workers = 1
    patience = 5

    data_dir = Path(data_dir)
    saving_dir = Path(saving_dir)
    saving_dir.mkdir(exist_ok = True, parents=True)

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        torchvision.transforms.Grayscale(num_output_channels=3),
        #transforms.RandomRotation(degrees=(0, 45)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])



    imgs = np.array([str(p) for p in data_dir.rglob("*/*")])
    labels = np.array([Path(p).parent.name for p in imgs])

    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(labels)
    classes = le.classes_
    labels = F.one_hot(torch.from_numpy(labels)).float()

    X_train_val, X_test, y_train_val, y_test = train_test_split(imgs, labels, test_size=test_size)

    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=test_size)


    n = int(X_train.shape[0]*sample)

    X_train = X_train[:n]
    y_train = y_train[:n]

    split_dict = {
        'train':{'images':X_train.tolist(),'labels':[int(np.argmax(l)) for l in y_train.tolist()]},
        'val':{'images':X_val.tolist(),'labels':[int(np.argmax(l)) for l in  y_val.tolist()]},
        'test':{'images':X_test.tolist(),'labels':[int(np.argmax(l)) for l in  y_test.tolist()]},
    }

    with open(saving_dir.joinpath('splits.json'),'w') as f:
        json.dump(split_dict,f)

    with open(saving_dir.joinpath('classes.json'),'w') as f:
        json.dump({'classes':classes.tolist()},f)


    train_dataset = TrainingDataset(X_train,y_train,transform = train_transform)
    val_dataset = TrainingDataset(X_val,y_val,transform = test_transform)
    test_dataset = TrainingDataset(X_test,y_test,transform = test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,collate_fn=my_collate,num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,collate_fn=my_collate,num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,collate_fn=my_collate,num_workers=num_workers)


    model = Classifier(
        output_dim = len(classes), 
        learning_rate = learning_rate,
        threshold = threshold
    )

    callbacks = [EarlyStopping(monitor="valid_loss",patience=patience, verbose = True)]

    trainer = pl.Trainer(
        #gpus=1,
        accelerator="auto",
        max_epochs = max_epochs,
        log_every_n_steps=100,
        callbacks = callbacks
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(dataloaders=test_loader)

    torch.save(model.state_dict(), saving_dir.joinpath('checkpoint.pth'))

if __name__ == '__main__':
    fire.Fire(main)