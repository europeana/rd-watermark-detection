import fire
import json
from pathlib import Path
import pandas as pd
from shutil import copyfile
import os

def main(**kwargs):

    labeled_dir = kwargs.get('labeled_dir')
    sample_dir = kwargs.get('sample_dir')
    labels = kwargs.get('labels')

    sample_dir = Path(sample_dir)

    labeled_dir = Path(labeled_dir)
    labeled_dir.mkdir(parents = True, exist_ok = True)

    df = pd.read_csv(labels)

    conv_dict = {
        'Watermarks':'watermark',
        'None':'no_watermark',
    }

    for label in conv_dict.values():
        labeled_dir.joinpath(label).mkdir(exist_ok = True, parents = True)

    #df['choice'] = df['choice'].apply(lambda x: conv_dict[x])

    num = 0
    for i,row in df.iterrows():

        name = Path(row['image']).name.replace('%5B','[').replace('%5D',']')
        path = sample_dir.joinpath(name)
  
        if not Path(path).is_file():
            continue

        dest_folder = labeled_dir.joinpath(row['choice'],Path(path).name)
        copyfile(path,dest_folder)
        os.remove(path)
        num += 1

    print(f'Total number of images tranferred: {num}')

    

if __name__ == "__main__":
    fire.Fire(main)