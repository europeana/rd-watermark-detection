import fastdup
import pandas as pd
from pathlib import Path
import fire

def main(**kwargs):

    data_dir = kwargs.get("data_dir")
    work_dir = kwargs.get("saving_dir")
    ccthreshold = kwargs.get("ccthreshold",0.9)
    threshold = kwargs.get("threshold",0.8)
    num_components = kwargs.get("num_components",15)

    path_list = []
    cat_list = []
    for cat in Path(data_dir).iterdir():
        for path in cat.iterdir():
            path_list.append(str(path))
            cat_list.append(cat.name)
            
    df_annot = pd.DataFrame({'filename':path_list,'label':cat_list})

    fd = fastdup.create(work_dir=work_dir, input_dir=data_dir) 
    fd.run(annotations=df_annot, ccthreshold=ccthreshold, threshold=threshold,overwrite=True)
    fd.vis.outliers_gallery()
    fd.vis.similarity_gallery() 
    fd.vis.duplicates_gallery()
    fd.vis.component_gallery(num_images=num_components)

if __name__ == "__main__":
    fire.Fire(main)