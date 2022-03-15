import fire
from pathlib import Path
import pandas as pd
import numpy as np
import json

def main(**kwargs):

    """
    python data-ops/format_predictions.py --metadata /home/jcejudo/projects/watermark_classification/data/watermarks.csv --predictions_path /home/jcejudo/projects/watermark_classification/results/iter_0/predictions.csv --saving_path /home/jcejudo/projects/watermark_classification/results/iter_0/predictions_labelstudio.json


    """

    saving_path = kwargs.get('saving_path')
    predictions_path = kwargs.get('predictions_path')
    metadata = kwargs.get('metadata')

    meta_df = pd.read_csv(metadata)

    def get_cat(x):
        return labels[np.argmax(x)]

    def get_max(x):
        return np.max(x)

    def fpath2id(x):
        return Path(x).with_suffix('').name.replace('[ph]','/')

    pred_df = pd.read_csv(predictions_path)
    labels = [col for col in pred_df.columns if col not in ['path','absdiff']]
    _df = pred_df[labels]
    pred_df['prediction'] = _df.apply(get_cat,axis=1)
    pred_df['confidence'] = _df.apply(get_max,axis=1)

    pred_df['europeana_id'] = pred_df['path'].apply(fpath2id)

    pred_df = pred_df.merge(meta_df)

    
    for col in ['absdiff']+labels:
        del pred_df[col]

    print(pred_df.head())

    pred_list = []

    for i,row in pred_df.iterrows():
        pred = {
            "data": {'image_url':row['image_url']},

            "predictions": [{
                "result": [
                {
                    "id": "result",
                    "type": "choices",
                    "from_name": "choice", "to_name": "image_url",
                    "value": {
                        "choices": [row['prediction']]
                        }
                }],
                "score": row['confidence']
            }]
            }

        pred_list.append(pred)

    with open(saving_path,'w') as f:
        json.dump(pred_list,f)




if __name__ == "__main__":
    fire.Fire(main)