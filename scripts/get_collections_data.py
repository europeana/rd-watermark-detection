import os
import time
import fire
import requests
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyeuropeana.apis as apis
import pyeuropeana.utils as utils

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
        query=f'edm_datasetName:{dataset_id}_*',
        rows=1,
        qf='TYPE:IMAGE',
        sort='random,europeana_id',
    )
    aggregator = response['items'][0]['provider'][0]
    provider = response['items'][0]['dataProvider'][0]
    return pd.Series([aggregator, provider], index=['aggregator', 'provider'])

def fetch_aggregators_and_providers(unique_datasets):
    """Use ThreadPoolExecutor to fetch data in parallel."""
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_dataset = {executor.submit(get_aggregator_provider, ds): ds for ds in unique_datasets}
        results = []
        for future in tqdm(as_completed(future_to_dataset), total=len(unique_datasets), desc="Fetching Data"):
            result = future.result()
            results.append(result)
    return pd.concat(results, axis=1).transpose()  # Adjust the axis to fit the DataFrame properly

def main(**kwargs):
    saving_path = kwargs.get("saving_path")
    df = fetch_collections()
    unique_datasets = list(df.dataset_name.unique())
    n_datasets = len(unique_datasets)
    print(f'Adding aggregators and providers to {n_datasets} datasets...')

    start = time.time()
    # Get data for each dataset
    results_df = fetch_aggregators_and_providers(unique_datasets)
    df = df.merge(results_df, left_on='dataset_name', right_index=True, how='left')

    df.to_csv(saving_path, index=False)

    end = time.time()
    dt = round((end - start) / 60, 2)
    print(f'\nFinished, it took {dt} minutes')

if __name__ == "__main__":
    fire.Fire(main)


