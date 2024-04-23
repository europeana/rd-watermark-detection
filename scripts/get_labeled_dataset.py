import os
import time
import fire
from pathlib import Path

import pandas as pd
import pyeuropeana.apis as apis
import pyeuropeana.utils as utils


def search_dataset(dataset_name, rows):
  dataset_id = dataset_name.split('_')[0]
  response = apis.search(
      query = f'edm_datasetName:{dataset_id}_*',
      qf = 'TYPE:IMAGE',
      media = True,
      rows = rows,
      sort = 'random,europeana_id'
      )
  return utils.search2df(response)

def main(**kwargs):
  input_path = kwargs.get("input_path")
  output_path = kwargs.get("output_path")

  print("Getting training dataset...")
  start = time.time()

  df = pd.read_csv(input_path)
  df = df.drop_duplicates(subset=['dataset_name'], keep = "last")

  cat_dict = {'watermarks':'Full','no_watermarks':'None'}
  rows_dict = {'watermarks':400,'no_watermarks':100}
  object_df = pd.DataFrame()
  for category in cat_dict.keys():
    print(category)
    cat_df = df.loc[df['watermarks'] == cat_dict[category]]
    total = cat_df.shape[0]
    print(f"Number of datasets: {total}")
    for i,dataset_name in enumerate(cat_df['dataset_name'].values[:]):
      response = search_dataset(dataset_name, rows_dict[category])
      response['category'] = category[:-1]
      object_df = pd.concat([object_df,response])
      if i>0 and i%5 == 0:
        print(f'{round(100*i/total,1)}%')

  object_df.to_csv(output_path,index = False)

  end = time.time()

  dt = (end-start)/60
  print(f"Finished, it took {dt} minutes")


if __name__ == "__main__":
  fire.Fire(main)