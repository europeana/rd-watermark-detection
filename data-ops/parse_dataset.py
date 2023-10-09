from pathlib import Path
import pandas as pd
import fire
import pyeuropeana.apis as apis

def id2url(id):
    try:
        data = apis.record(id)
        return data['object']['aggregations'][0]['edmObject']
    except:
        return None

def main(**kwargs):

    # python data-ops/parse_dataset.py --dataset_path /home/jcejudo/projects/watermark_classification/data/labeled --output_path /home/jcejudo/projects/watermark_classification/data/labeled.csv

    dataset_path = kwargs.get("dataset_path")
    output_path = kwargs.get("output_path")

    dataset_path = Path(dataset_path)

    cat_list = []
    id_list = []
    for cat_path in dataset_path.iterdir():
        id_list += [fpath.with_suffix("").name.replace("[ph]",'/') for fpath in cat_path.iterdir()]
        cat_list += [cat_path.name for fname in cat_path.iterdir() ]

    df = pd.DataFrame({'id':id_list,'category':cat_list})
    df['uri'] = df['id'].apply(lambda x: "https://www.europeana.eu/en/item"+x)
    df['image_url'] = df['id'].apply(lambda x: id2url(x))
    df.to_csv(output_path,index = False)



if __name__ == "__main__":
    fire.Fire(main)
