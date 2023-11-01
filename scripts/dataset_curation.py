import fastdup
import pandas as pd
from pathlib import Path
import fire
import torch
from PIL import Image
from cleanlab import Datalab
import matplotlib.pyplot as plt

def run_fastdup(**kwargs):

    data_dir = kwargs.get("data_dir")
    work_dir = kwargs.get("saving_dir")
    ccthreshold = kwargs.get("ccthreshold",0.9)
    threshold = kwargs.get("threshold",0.8)
    num_components = kwargs.get("num_components",15)

    path_list = []
    cat_list = []
    for cat in Path(data_dir).iterdir():
        for path in cat.iterdir():
            path_list.append(str(path))
            cat_list.append(cat.name)
            
    df_annot = pd.DataFrame({'filename':path_list,'label':cat_list})

    fd = fastdup.create(work_dir=work_dir, input_dir=data_dir) 
    fd.run(annotations=df_annot, ccthreshold=ccthreshold, threshold=threshold,overwrite=True)
    fd.vis.outliers_gallery()
    fd.vis.similarity_gallery() 
    fd.vis.duplicates_gallery()
    fd.vis.component_gallery(num_images=num_components)

def run_cleanlab(**kwargs):
    results_dir = kwargs.get('results_dir')
    results_dir = Path(results_dir)

    features = torch.load(results_dir.joinpath('features.pt'))
    pred_probs = torch.load(results_dir.joinpath('pred_probs.pt'))
    dataset = torch.load(results_dir.joinpath('dataset.pt'))
    print(dataset)
    imgs = dataset['image']

    lab = Datalab(data=dataset, label_name="label", image_key="image")
    lab.find_issues(features=features, pred_probs=pred_probs)
    lab.report()

def main(*args,**kwargs):
    arg = args[0]
    if arg == 'fastdup':
        run_fastdup(**kwargs)
    elif arg == 'cleanlab':
        run_cleanlab(**kwargs)




if __name__ == "__main__":
    fire.Fire(main)