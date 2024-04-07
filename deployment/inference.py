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
from sklearn.model_selection import train_test_split, StratifiedKFold
import json
from tqdm import tqdm
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score,accuracy_score, recall_score
import matplotlib.pyplot as plt
from shutil import copyfile

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from torch.utils.data import Dataset, DataLoader
from PIL import Image

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from pytorch_lightning.loggers import TensorBoardLogger



def fpath2id(x):
    return Path(x).with_suffix('').name.replace('[ph]','/')

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


class InferenceDataset(Dataset):
  
    def __init__(self, imgs, transform=None):
        self.transform = transform
        self.img_list = imgs

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_list[idx]

        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except:
            return None
        return img_path,image


class Classifier(pl.LightningModule): 
  
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        output_dim = kwargs.get('output_dim')
        learning_rate = kwargs.get('learning_rate')
        self.threshold = kwargs.get('threshold')

        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, output_dim)
        self.embedding = nn.Sequential(*list(self.model.children())[:-1])
        self.sm = nn.Softmax(dim=1)
        self.accuracy = torchmetrics.Accuracy(task='binary')

    def forward(self, x):
        out = self.model(x)
        out = self.sm(out)
        return out

    def embeddings(self, x):
        out = self.embedding(x)
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

# Method for computing out-of-sample embeddings
def compute_embeddings(model, testloader,device):
    embeddings_list = []

    with torch.no_grad():
        for data in tqdm(testloader):
            images, labels = data[0].to(device), data[1].to(device)

            embeddings = model.embeddings(images)
            embeddings_list.append(embeddings.cpu())

    return torch.vstack(embeddings_list)


# Method for computing out-of-sample predicted probabilities
def compute_pred_probs(model, testloader,device):
    pred_probs_list = []

    with torch.no_grad():
        for data in tqdm(testloader):
            images, labels = data[0].to(device), data[1].to(device)

            outputs = model(images)
            pred_probs_list.append(outputs.cpu())

    return torch.vstack(pred_probs_list)



def predict(**kwargs):

    results_path = kwargs.get('results_path')
    saving_path = kwargs.get('saving_path')
    input = kwargs.get('input')
    threshold = kwargs.get('threshold',0.5)

    sample = kwargs.get('sample',1.0)
    batch_size = kwargs.get('batch_size',4)
    metadata = kwargs.get('metadata')

    num_workers = 6
    
    meta_df = pd.read_csv(metadata)

    results_path = Path(results_path)
    input = Path(input)

    with open(results_path.joinpath('classes.json'),'r') as f:
        classes = json.load(f)['classes']

    n_classes = len(classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Loading model...')

    model = Classifier(
        output_dim = n_classes, 
        learning_rate = 0.0,
        threshold = threshold
    )

    model.load_state_dict(torch.load(results_path.joinpath('checkpoint.pth')))
    print('Model loaded')

    print(f'Device: {device}')

    model = model.to(device)

    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    print('Loading dataset...')

    image_arr = np.array([str(path) for path in input.rglob('*.jpg')])
    n = image_arr.shape[0]
    index = np.random.choice(n, int(sample*n), replace=False)  
    image_arr = image_arr[index]
    df = pd.DataFrame({'path':image_arr})
    df['europeana_id'] = df['path'].apply(fpath2id)

    ids = set(meta_df['europeana_id'])  
    df = df[df['europeana_id'].isin(ids)]
    image_arr = df['path'].values

    print('Dataset loaded')

    print(f'Predicting on {image_arr.shape[0]} images...')

    train_dataset = InferenceDataset(image_arr,transform = transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,collate_fn=my_collate,num_workers=num_workers)

    conf_list = []
    path_list = []
    prediction_list = []
    with torch.no_grad():
        for paths,batch in tqdm(train_loader):
            batch = batch.to(device)
            outputs = model(batch).cpu()
            prediction_list += [classes[i] for i in torch.argmax(outputs,axis=1)]
            conf_list.append(outputs)
            path_list.append(paths)

    print('Finished predicting')

    output = torch.Tensor(len(train_loader)*batch_size, n_classes).cpu()
    output = torch.cat(conf_list, out=output)

    path_list = [list(t) for t in path_list]
    path_list = [item for sublist in path_list for item in sublist]
    
    conf_dict = {'path':path_list}
    conf_dict.update({cat:output[:,i] for i,cat in enumerate(classes)})

    def absdiff(x):
        return np.abs(x[1]-x[2])

    df = pd.DataFrame(conf_dict)
    df['prediction'] = prediction_list
    df['absdiff'] = df.apply(absdiff, axis=1)


    df = df.sort_values(by=['absdiff'],ascending = False)

    df['europeana_id'] = df['path'].apply(fpath2id)

    df = df.merge(meta_df)
    df = df.drop_duplicates(subset=['path'])
    #df = df.head(n_predictions) 
    df.to_csv(saving_path,index=False)
    print(df.shape)

    # saving sample

    #if sample_path:
    #    sample_path = Path(sample_path)
    #    sample_path.mkdir(parents = True, exist_ok = True)

    #    for path in df['path'].values:
    #        copyfile(path, sample_path.joinpath(Path(path).name))

    print('Finished')

    

def main(*args,**kwargs):
    predict(**kwargs)


if __name__ == '__main__':
    fire.Fire(main)