import os
import sys
import fire
from shutil import copyfile

from pathlib import Path
import numpy as np
import pandas as pd

import json

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image
import torch


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from train import Classifier, my_collate

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

def fpath2id(x):
    return Path(x).with_suffix('').name.replace('[ph]','/')

def main(**kwargs):

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


if __name__ == "__main__":
    fire.Fire(main)


