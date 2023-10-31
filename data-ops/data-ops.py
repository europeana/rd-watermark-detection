import fire
import pandas as pd
import json
from pathlib import Path
from shutil import copyfile
import os
from tqdm import tqdm

import pyeuropeana.apis as apis
import pyeuropeana.utils as utils

def harvest_watermark(**kwargs):
  n_per_dataset = kwargs.get('n_per_dataset',500)
  saving_path = kwargs.get('saving_path')
  query_list = kwargs.get('query_list')

  df = pd.DataFrame()

  #harvest watermarks
  for i,query in enumerate(query_list):
    print(f'{i} out of {len(query_list)}')
    response = apis.search(
        qf = query, 
        sort = 'random,europeana_id',
        rows = n_per_dataset, 
        )
    _df = utils.search2df(response)
    if _df is None:
      continue
    _df['query'] = query
    _df['category'] = 'watermark'
    df = df.append(_df)

  df = df.drop_duplicates(subset=['europeana_id'])
  return df

def harvest_normal(**kwargs):

  n = kwargs.get('n',100)

  # harvest without watermark
  response = apis.search(
      query = '*',
      qf = 'TYPE:IMAGE', 
      media = True,
      rows = n, 
      sort = 'random,europeana_id'
      )
  df = utils.search2df(response)
  df['query'] = '*'
  df['category'] = 'no_watermark'
  df = df.drop_duplicates(subset=['europeana_id'])
  return df

  
def harvest_data(**kwargs):

    saving_path = kwargs.get('saving_path')
    labeled_path = kwargs.get('labeled_path')
    n_per_dataset = kwargs.get('n_per_dataset',150)
    datasets_path = kwargs.get('datasets_path')

    with open(datasets_path, 'r') as f:
      query_list = json.load(f)['query_list']

    print('Getting watermarks ...')

    watermark_df = harvest_watermark(
        n_per_dataset = n_per_dataset,
        query_list = query_list,
    )

    print('Getting no-watermarks ...')

    normal_df = harvest_normal(
        n = watermark_df.shape[0],
    )

    df = pd.concat([watermark_df,normal_df])
    print(df.shape)

    # remove objects present in the labeled dataset
    if labeled_path:
      labeled_df = pd.read_csv(labeled_path)
      df = df.loc[df.apply(lambda x: x['europeana_id'] not in labeled_df['id'].values,axis=1)]

    df.to_csv(saving_path,index=False)

    print('Finished')

def download_images(df,saving_dir,n_images = 10):

  watermark_dir = saving_dir.joinpath('watermark')
  watermark_dir.mkdir(parents=True,exist_ok=True)

  no_watermark_dir = saving_dir.joinpath('no_watermark')
  no_watermark_dir.mkdir(parents=True,exist_ok=True)

  print_every = 100

  df = df.sample(frac = 1).reset_index(drop=True)

  print(df['category'].values[:20])

  for i,row in df.iterrows():
    if i > n_images:
      break
    try:
      img = utils.url2img(row['image_url'])
    except:
      img = None
    if not img:
      continue
    

    fname = row['europeana_id'].replace('/','[ph]')+'.jpg'

    img.save(saving_dir.joinpath(row['category'],fname))

    if i % print_every == 0:
      print(i)

def download(**kwargs):
    saving_dir = kwargs.get('saving_dir')
    input = kwargs.get('input')

    saving_dir = Path(saving_dir)

    saving_dir.mkdir(parents=True,exist_ok=True)

    df = pd.read_csv(input)
    print(f'Downloading {df.shape[0]} images')
    download_images(df,saving_dir,n_images = 1e6)
    print('Finished')

def move_labeled(**kwargs):

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

def id2url(id):
    try:
        data = apis.record(id)
        return data['object']['aggregations'][0]['edmObject']
    except:
        return None

def parse_dataset(**kwargs):

    # python data-ops/parse_dataset.py --dataset_path /home/jcejudo/projects/watermark_classification/data/labeled --output_path /home/jcejudo/projects/watermark_classification/data/labeled.csv

    dataset_path = kwargs.get("dataset_path")
    output_path = kwargs.get("output_path")


    dataset_path = Path(dataset_path)

    tqdm.pandas()

    cat_list = []
    id_list = []
    for cat_path in dataset_path.iterdir():
        id_list += [fpath.with_suffix("").name.replace("[ph]",'/') for fpath in cat_path.iterdir()]
        cat_list += [cat_path.name for fname in cat_path.iterdir() ]

    df = pd.DataFrame({'id':id_list,'category':cat_list})
    df['uri'] = df['id'].apply(lambda x: "https://www.europeana.eu/en/item"+x)
    df['image_url'] = df['id'].apply(lambda x: id2url(x))
    df.to_csv(output_path,index = False)
    print('Finished')


def main(*args,**kwargs):
    arg = args[0]
    if arg == 'harvest_data':
        harvest_data(**kwargs)
    elif arg == 'download':
        download(**kwargs)
    elif arg == 'move_labeleds':
        move_labeled(**kwargs)
    elif arg == 'parse_dataset':
        parse_dataset(**kwargs)

      
if __name__ == "__main__":
    fire.Fire(main)