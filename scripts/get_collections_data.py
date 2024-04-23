import os
import time
import fire
import requests
import pandas as pd
from tqdm import tqdm

import pyeuropeana.apis as apis
import pyeuropeana.utils as utils

os.environ['EUROPEANA_API_KEY'] = 'api2demo'

def fetch_collections():
    """Fetch collection data from the given URL."""
    wskey = os.getenv("EUROPEANA_API_KEY")
    url = f'https://api.europeana.eu/record/v2/search.json?wskey={wskey}&qf=TYPE:%22IMAGE%22&query=*&rows=0&profile=facets&facet=edm_datasetName&f.edm_datasetName.facet.limit=10000'
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        df = pd.DataFrame(response.json()['facets'][0]['fields'])
        df = df.rename(columns={"label": "dataset_name","count":"size_dataset"})
        return df
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return pd.DataFrame()

def get_aggregator_provider(dataset):
    dataset_id = dataset.split('_')[0]
    response = apis.search(
        query = f'edm_datasetName:{dataset_id}_*',
        rows = 1,
        qf = 'TYPE:IMAGE',
        sort = 'random,europeana_id',
        )
    aggregator = response['items'][0]['provider'][0]
    provider = response['items'][0]['dataProvider'][0]
    return pd.Series([aggregator, provider])

def main(**kwargs):
    #saving_path = "/content/collections.csv"
    saving_path = kwargs.get("saving_path")
    df = fetch_collections()
    #df = df.head(5)
    unique_datasets = list(df.dataset_name.unique())
    n_datasets = len(unique_datasets)
    print(f'Adding aggregators and providers to {n_datasets} datasets...')
    start = time.time()
    tqdm.pandas(desc="Processing Datasets")
    df[['aggregator', 'provider']] = df['dataset_name'].progress_apply(get_aggregator_provider)
    df.to_csv(saving_path,index = False)
    end = time.time()
    dt = round((end-start)/60,2)
    print(f'\nFinished, it took {dt} minutes')

if __name__ == "__main__":
    fire.Fire(main)


