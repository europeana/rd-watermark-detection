import os
import sys
import fire

from pathlib import Path
import numpy as np
import pandas as pd

import json

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.nn import functional as F

from PIL import Image
import torch

import seaborn as sn

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score,accuracy_score, recall_score


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from train import Classifier, my_collate, TrainingDataset
from predict import InferenceDataset

def main(**kwargs):

    results_path = kwargs.get('results_path')
    saving_path = kwargs.get('saving_path')
    threshold = kwargs.get('threshold',0.5)
    sample = kwargs.get('sample',1.0)
    batch_size = kwargs.get('batch_size',4)

    results_path = Path(results_path)
    saving_path = Path(saving_path)

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

    with open(results_path.joinpath('splits.json'),'r') as f:
        test_split = json.load(f)['test']

    image_arr = np.array(test_split['images'])
    label_arr = np.array([classes[i] for i in test_split['labels']])

    print(f'evaluating on {image_arr.shape[0]} images')

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

    prediction_arr = np.array(prediction_list)
    path_list = [list(t) for t in path_list]
    path_list = [item for sublist in path_list for item in sublist]

    df = pd.DataFrame({
        'path':path_list,
        'predictions':prediction_arr.tolist(),
        'labels':label_arr.tolist(),
        })

    df = df.loc[df['predictions'] == df['labels']]
    df.to_csv(saving_path.joinpath('misclassifications.csv'),index=False)

    print('Calculating metrics')


    cm = confusion_matrix(label_arr,prediction_arr)
    cm = cm.tolist()

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
    sn.heatmap(df_cm, annot=True,cmap='Blues', fmt='g')
    fig.savefig(saving_path.joinpath('confusion_matrix.jpg'))

    # save metris in json
    with open(saving_path.joinpath('metrics.json'),'w') as f:
        json.dump(metrics_dict,f)

    print('Finished')



if __name__ == "__main__":
    fire.Fire(main)


