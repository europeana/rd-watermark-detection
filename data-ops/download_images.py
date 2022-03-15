import fire
from pathlib import Path
import pandas as pd

import pyeuropeana.utils as utils

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


def main(**kwargs):
    saving_dir = kwargs.get('saving_dir')
    input = kwargs.get('input')

    saving_dir = Path(saving_dir)

    saving_dir.mkdir(parents=True,exist_ok=True)

    df = pd.read_csv(input)
    print(f'Downloading {df.shape[0]} images')
    download_images(df,saving_dir,n_images = 1e6)

if __name__ == '__main__':
    fire.Fire(main)