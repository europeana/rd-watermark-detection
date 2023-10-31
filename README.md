# Detection of watermarks in images

## Setting up environment

docker-compose up -d

docker-compose exec label_studio bash


https://github.com/HumanSignal/label-studio/issues/3987

Create docker image

```
docker build . -t watermark_image
```

Run docker container

docker run --gpus all --env LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true --env LOCAL_FILES_SERVING_ENABLED=true --env LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/files --env LOCAL_FILES_DOCUMENT_ROOT=/label-studio/files -p 8092:8092 -v /home/jcejudo/projects/watermark_classification:/output -v /home/jcejudo/projects/watermark_classification:/label-studio/files -v $(pwd)/mydata:/label-studio/data -v $(pwd):/code -it watermark_image:latest

label-studio -p 8092

docker run --gpus all -p 8092:8092 -v /home/jcejudo/projects/watermark_classification:/output -v /home/jcejudo/projects/watermark_classification:/label-studio/files -v $(pwd)/mydata:/label-studio/data -v $(pwd):/code -it watermark_image:latest




## Data acquisition

```
nohup python3 data-ops/harvest_data.py --datasets_path /output/data/new_datasets.json --n_per_dataset 200 --saving_path /output/data/unlabeled.csv --labeled_path /output/data/labeled_4312.csv &> /output/results/data_harvesting.out &
```

```
nohup python3 data-ops/download_images.py --input /output/data/unlabeled.csv --saving_dir /output/data/unlabeled &> /output/results/download_images.out &
```


## Model training

```
nohup python3 machine-learning/train.py --batch_size 16 --data_dir /output/data/labeled_4312 --saving_dir /output/results/iter_6 --max_epochs 20 --sample 1.0 &> /output/results/training.out &
```

```
python3 machine-learning/evaluate.py --results_path /output/results/iter_6 --saving_path /output/results/iter_6
```

## Predict

```
python3 machine-learning/predict.py --input /output/data/unlabeled --results_path /output/results/iter_6 --metadata /output/data/unlabeled.csv --saving_path /output/results/iter_6/predictions.csv --mode uncertain --n_predictions 800 --sample 1.0 --batch_size 64 --sample_path /output/results/iter_6/sample
```

## Annotate with Label-Studio

to do: add config file

```
label-studio -p 8092

```

to do: add path to sampled images in labelstudio interface

import predictions to labelstudio, label, export annotations

moving annotations to labeled and removing from unlabeled images

to do

```
python3 data-ops/move_labeled.py --sample_dir /output/results/iter_6/sample --labeled_dir /output/data/labeled --labels '/output/results/iter_6/project-1-at-2023-10-31-14-05-97816efa.csv'


```



parse labeled dataset

to do: include error message if api key not detected

```nohup python3 data-ops/parse_dataset.py --dataset_path /output/data/labeled --output_path /output/data/labeled.csv &> /output/results/parsing_labeled.out &```


























