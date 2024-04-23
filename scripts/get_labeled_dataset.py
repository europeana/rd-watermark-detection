import os
import time
import fire
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import pyeuropeana.apis as apis
import pyeuropeana.utils as utils
from concurrent.futures import ThreadPoolExecutor, as_completed



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

def search_and_append(results, dataset_name, category, rows_dict):
    response = search_dataset(dataset_name, rows_dict[category])
    response['category'] = category[:-1] 
    results.append(response)

def main(**kwargs):
    input_path = kwargs.get("input_path")
    output_path = kwargs.get("output_path")

    print("Getting training dataset...")
    start = time.time()

    df = pd.read_csv(input_path)
    df = df.drop_duplicates(subset=['dataset_name'], keep="last")

    cat_dict = {'watermarks': 'Full', 'no_watermarks': 'None'}
    rows_dict = {'watermarks': 400, 'no_watermarks': 100}

    object_df = pd.DataFrame()
    results = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_ds = {executor.submit(search_and_append, results, dataset_name, category, rows_dict): dataset_name 
                        for category in cat_dict.keys() 
                        for dataset_name in df.loc[df['watermarks'] == cat_dict[category]]['dataset_name'].values}

        for future in tqdm(as_completed(future_to_ds), total=len(future_to_ds), desc="Searching Datasets"):
            dataset_name = future_to_ds[future]
            try:
                future.result() 
            except Exception as exc:
                print(f'{dataset_name} generated an exception: {exc}')

    object_df = pd.concat(results, ignore_index=True)
    object_df.to_csv(output_path, index=False)

    end = time.time()
    dt = (end - start) / 60
    print(f"Finished, it took {dt:.2f} minutes")



if __name__ == "__main__":
  fire.Fire(main)