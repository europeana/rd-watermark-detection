import fire
import pandas as pd

import pyeuropeana.apis as apis
import pyeuropeana.utils as utils

def harvest_watermark(**kwargs):
  n_per_dataset = kwargs.get('n_per_dataset',500)
  saving_path = kwargs.get('saving_path')

  query_list = [
    'edm_datasetName:481_*', # https://metis-publish-portal.eanadev.org/en/search?query=edm_datasetName%3A481_%2A
    'edm_datasetName:472_*', # https://metis-publish-portal.eanadev.org/en/search?query=edm_datasetName%3A472_%2A
    'edm_datasetName:473_*', # https://metis-publish-portal.eanadev.org/en/search?query=edm_datasetName%3A473_%2A
    'edm_datasetName:470_*', # https://metis-publish-portal.eanadev.org/en/search?query=edm_datasetName%3A470_%2A
    'edm_datasetName:482_*', # https://metis-publish-portal.eanadev.org/en/search?query=edm_datasetName%3A482_%2A
    'edm_datasetName:475_*', # https://metis-publish-portal.eanadev.org/en/search?query=edm_datasetName%3A475_%2A
    'edm_datasetName:476_*', # https://metis-publish-portal.eanadev.org/en/search?query=edm_datasetName%3A476_%2A
    'edm_datasetName:477_*', # https://metis-publish-portal.eanadev.org/en/search?query=edm_datasetName%3A477_%2A
    'edm_datasetName:478_*', # https://metis-publish-portal.eanadev.org/en/search?query=edm_datasetName%3A478_%2A
    'edm_datasetName:479_*', # https://metis-publish-portal.eanadev.org/en/search?query=edm_datasetName%3A479_%2A
    'edm_datasetName:484_*', # https://metis-publish-portal.eanadev.org/en/search?query=edm_datasetName%3A484_%2A
    'edm_datasetName:486_*', # https://metis-publish-portal.eanadev.org/en/search?query=edm_datasetName%3A486_%2A
    'edm_datasetName:488_*', # https://metis-publish-portal.eanadev.org/en/search?query=edm_datasetName%3A488_%2A
    'edm_datasetName:2023019_*', # https://metis-publish-portal.eanadev.org/en/search?query=edm_datasetName%3A2023019_%2A
    'edm_datasetName:2048217_*', # https://metis-publish-portal.eanadev.org/en/search?query=edm_datasetName%3A2048217_%2A
    'edm_datasetName:2048202_*', # https://metis-publish-portal.eanadev.org/en/search?query=edm_datasetName%3A2048202_%2A
    'edm_datasetName:2022713*', # https://www.europeana.eu/en/search?page=1&view=grid&query=edm_datasetName%3A2022713%2A
  ]


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

  
def main(**kwargs):

    saving_path = kwargs.get('saving_path')

    print('Getting watermarks ...')

    watermark_df = harvest_watermark(
        n_per_dataset = 100,
    )

    print('Getting no-watermarks ...')

    normal_df = harvest_normal(
        n = watermark_df.shape[0],
    )

    df = pd.concat([watermark_df,normal_df])
    print(df.shape)
    df.to_csv(saving_path,index=False)

    print('Finished')

if __name__ == "__main__":
    fire.Fire(main)