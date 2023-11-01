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

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
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

def train(**kwargs):

    data_dir = kwargs.get('data_dir')
    saving_dir = kwargs.get('saving_dir')
    max_epochs = kwargs.get('max_epochs',10)
    sample = kwargs.get('sample',1.0)
    test_size = kwargs.get('test_size',0.2)
    batch_size = kwargs.get('batch_size',16)
    learning_rate = kwargs.get('learning_rate',1e-4)
    threshold = kwargs.get('threshold',0.5)
    num_workers = kwargs.get('num_workers',1)
    patience = kwargs.get('patience',5)
    crossvalidation = kwargs.get('crossvalidation',False)
    K = kwargs.get('K',5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = Path(data_dir)
    saving_dir = Path(saving_dir)
    saving_dir.mkdir(exist_ok = True, parents=True)

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([transforms.Grayscale(num_output_channels=3)], p=0.25),
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

    # n = int(imgs.shape[0]*sample)
    # imgs = imgs[:n]
    # labels = labels[:n]

    le = preprocessing.LabelEncoder()
    _labels = le.fit_transform(labels)
    classes = le.classes_
    labels = F.one_hot(torch.from_numpy(_labels)).float()

    # Create k splits of the dataset

    kfold = StratifiedKFold(n_splits=K, shuffle=True, random_state=0)
    splits = kfold.split(imgs, _labels)

    train_id_list, test_id_list = [], []

    for fold, (train_ids, test_ids) in enumerate(splits):
        train_id_list.append(train_ids)
        test_id_list.append(test_ids)


    pred_probs_list, embeddings_list = [], []

    for i in range(K):

        if crossvalidation == False and i>0:
            break

        split_dir = saving_dir.joinpath(f'split_{i+1}')
        split_dir.mkdir(exist_ok = True, parents=True)

        print(f"\nTraining on fold: {i+1} ...")

        # Create train and test sets and corresponding dataloaders

        X_train_val = imgs[train_id_list[i]]
        y_train_val = labels[train_id_list[i]]

        # Create validation split

        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=1.0/K)

        X_test = imgs[test_id_list[i]]
        y_test = labels[test_id_list[i]]

        n = int(X_train.shape[0]*sample)
        X_train = X_train[:n]
        y_train = y_train[:n]

        split_dict = {
            'train':{'images':X_train.tolist(),'labels':[int(np.argmax(l)) for l in y_train.tolist()]},
            'val':{'images':X_val.tolist(),'labels':[int(np.argmax(l)) for l in  y_val.tolist()]},
            'test':{'images':X_test.tolist(),'labels':[int(np.argmax(l)) for l in  y_test.tolist()]},
        }

        with open(split_dir.joinpath('splits.json'),'w') as f:
            json.dump(split_dict,f)

        with open(split_dir.joinpath('classes.json'),'w') as f:
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

        logger = TensorBoardLogger(name="tensorboard_logs", save_dir = split_dir)

        trainer = pl.Trainer(
            accelerator="auto",
            max_epochs = max_epochs,
            log_every_n_steps=100,
            callbacks = callbacks,
            logger = logger
        )

        trainer.fit(model, train_loader, val_loader)
        trainer.test(dataloaders=test_loader)
        torch.save(model.state_dict(), split_dir.joinpath('checkpoint.pth'))


        # Evaluation

        model.eval()

        with open(split_dir.joinpath('splits.json'),'r') as f:
            test_split = json.load(f)['test']

        image_arr = np.array(test_split['images'])
        label_arr = np.array([classes[i] for i in test_split['labels']])

        print(f'evaluating on {image_arr.shape[0]} images')

        inference_dataset = InferenceDataset(image_arr,transform = test_transform)
        inference_loader = DataLoader(inference_dataset, batch_size=batch_size,collate_fn=my_collate)

        conf_list = []
        path_list = []
        prediction_list = []
        with torch.no_grad():
            for paths,batch in inference_loader:
                outputs = model(batch)
                prediction_list += [classes[i] for i in torch.argmax(outputs,axis=1)]
                conf_list.append(outputs)
                path_list.append(paths)

        prediction_arr = np.array(prediction_list)
        path_list = [list(t) for t in path_list]
        path_list = [item for sublist in path_list for item in sublist]

        df = pd.DataFrame({
            'path':path_list,
            'predictions':prediction_arr.tolist(),
            'labels':label_arr.tolist(),
            })
        df = df.loc[df['predictions'] != df['labels']]
        df.to_csv(split_dir.joinpath('misclassifications.csv'),index=False)

        print('Calculating metrics')

        cm = confusion_matrix(label_arr,prediction_arr).tolist()
        precision = precision_score(label_arr,prediction_arr,average="binary",pos_label="watermark")
        recall = recall_score(label_arr,prediction_arr,average="binary",pos_label="watermark")
        accuracy = accuracy_score(label_arr,prediction_arr)

        metrics_dict = {
            'accuracy':accuracy,
            'precision':precision,
            'recall':recall,
            'cm':cm,
        }

        # plot confusion matrix
        fig,ax = plt.subplots()
        df_cm = pd.DataFrame(cm, index = classes,
                    columns = classes)
        ax = sn.heatmap(df_cm, annot=True,cmap='Blues', fmt='g')
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        fig.savefig(split_dir.joinpath('confusion_matrix.jpg'))

        # save metris in json
        with open(split_dir.joinpath('metrics.json'),'w') as f:
            json.dump(metrics_dict,f)



        # Calculate embeddings
        model = model.to(device)

        # Compute out-of-sample embeddings
        print("Computing feature embeddings ...")
        fold_embeddings = compute_embeddings(model, test_loader,device)
        embeddings_list.append(fold_embeddings)

        print("Computing predicted probabilities ...")
        # Compute out-of-sample predicted probabilities
        fold_pred_probs = compute_pred_probs(model, test_loader,device)
        pred_probs_list.append(fold_pred_probs)

    print("Finished Training")


    # Combine embeddings and predicted probabilities from each fold
    features = torch.vstack(embeddings_list).numpy()
    features = np.squeeze(features)

    logits = torch.vstack(pred_probs_list)
    pred_probs = logits.numpy()

    indices = np.hstack(test_id_list)

    imgs = imgs[indices]
    _labels = _labels[indices]
    dataset = {'image':imgs,'label':_labels}

    torch.save(features, saving_dir.joinpath('features.pt'))
    torch.save(dataset, saving_dir.joinpath('dataset.pt'))
    torch.save(pred_probs, saving_dir.joinpath('pred_probs.pt'))

    print("Finished")

def predict(**kwargs):

    results_path = kwargs.get('results_path')
    saving_path = kwargs.get('saving_path')
    input = kwargs.get('input')
    threshold = kwargs.get('threshold',0.5)

    sample = kwargs.get('sample',1.0)
    batch_size = kwargs.get('batch_size',4)
    metadata = kwargs.get('metadata')
    n_predictions = kwargs.get('n_predictions',200)

    mode = kwargs.get('mode','uncertain')
    sample_path = kwargs.get('sample_path')

    meta_df = pd.read_csv(metadata)

    results_path = Path(results_path)
    input = Path(input)

    with open(results_path.joinpath('classes.json'),'r') as f:
        classes = json.load(f)['classes']

    n_classes = len(classes)

    model = Classifier(
        output_dim = n_classes, 
        learning_rate = 0.0,
        threshold = threshold
    )

    model.load_state_dict(torch.load(results_path.joinpath('checkpoint.pth')))

    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image_arr = np.array([str(path) for path in input.rglob('*.jpg')])
    # filter images in meta df
    id_arr = [fpath2id(path) for path in image_arr]
    idx = [True if id in meta_df['europeana_id'].values else False for id in id_arr]
    image_arr = image_arr[idx]
    # sample images for predicting
    idx = np.random.randint(image_arr.shape[0],size = int(sample*image_arr.shape[0]))
    image_arr = image_arr[idx]

    print(f'predicting on {image_arr.shape[0]} images')

    train_dataset = InferenceDataset(image_arr,transform = transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,collate_fn=my_collate)

    conf_list = []
    path_list = []
    prediction_list = []
    with torch.no_grad():
        for paths,batch in train_loader:
            outputs = model(batch)
            prediction_list += [classes[i] for i in torch.argmax(outputs,axis=1)]
            conf_list.append(outputs)
            path_list.append(paths)

    print('Finished predicting')

    output = torch.Tensor(len(train_loader)*batch_size, n_classes)
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

    if mode == 'uncertain': # for getting the most uncertain results
        df = df.sort_values(by=['absdiff'])
    else:
        df = df.sort_values(by=['absdiff'],ascending = False)

    df['europeana_id'] = df['path'].apply(fpath2id)

    df = df.merge(meta_df)
    df = df.drop_duplicates(subset=['path'])
    df = df.head(n_predictions) 
    df.to_csv(saving_path,index=False)
    print(df.shape)

    # saving sample

    sample_path = Path(sample_path)
    sample_path.mkdir(parents = True, exist_ok = True)

    for path in df['path'].values:
        copyfile(path, sample_path.joinpath(Path(path).name))

    print('Finished')

    

def main(*args,**kwargs):
    arg = args[0]
    if arg == 'train':
        train(**kwargs)
    elif arg == 'predict':
        predict(**kwargs)


if __name__ == '__main__':
    fire.Fire(main)