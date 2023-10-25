# Detection of watermarks in images

## Setting up environment

Create docker image

```
docker build . -t watermark_image
```

Run docker container

```
docker run --gpus all -p 8090:8090 -v /home/jcejudo/projects/watermark_classification:/output -v $(pwd):/code -it watermark_image:latest
```


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
python3 machine-learning/predict.py --input /output/data/unlabeled --results_path /output/results/iter_6 --metadata /output/data/unlabeled.csv --saving_path /output/results/iter_6/predictions.csv --mode uncertain --n_predictions 700 --sample 1.0 --batch_size 64
```

## Annotate with Label-Studio

```
label-studio -p 8090
```


import predictions to labelstudio, label, export annotations


moving annotations to labeled and removing from unlabeled images

to do

```
python data-ops/move_labeled.py --unlabeled_dir /home/jcejudo/projects/watermark_classification/data/unlabeled --labeled_dir /home/jcejudo/projects/watermark_classification/data/labeled --labels '/home/jcejudo/projects/watermark_classification/results/iter_5/project-15-at-2023-10-09-18-59-31d43bb2.csv'
```



parse labeled dataset

to do: include error message if api key not detected

```nohup python3 data-ops/parse_dataset.py --dataset_path /output/data/labeled --output_path /output/data/labeled.csv &> /output/results/parsing_labeled.out &```


























