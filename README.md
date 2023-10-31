# Detection of watermarks in images

docker-compose up -d

Services:
machine_learning
label_studio


## Data acquisition

docker-compose exec machine_learning bash

```
nohup python3 data-ops/data-ops.py harvest_data --datasets_path /storage/data/new_datasets.json --n_per_dataset 200 --saving_path /storage/data/unlabeled.csv --labeled_path /storage/data/labeled_4312.csv &> /storage/results/data_harvesting.out &
```

```
nohup python3 data-ops/data-ops.py download --input /storage/data/unlabeled.csv --saving_dir /storage/data/unlabeled &> /storage/results/download_images.out &
```


## Model training

to do: add crossvalidation
include evaluation in training
merge the three files into a single one

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

Access the container of label studio

```
docker-compose exec label_studio bash
```

Start watermark_detection project

```
label-studio init watermark_detection --label-config /code/labelstudio-config.xml
```

Start interface

```
label-studio start watermark_detection -p 8093
```

Go to the watermark_detection project -> Settings -> Cloud Storage -> Source storage -> Local storage

Annotate and export as CSV

moving annotations to labeled and removing from unlabeled images

```
python3 data-ops/data-ops.py move_labeled --sample_dir /output/results/iter_6/sample --labeled_dir /output/data/labeled --labels '/output/results/iter_6/project-1-at-2023-10-31-14-05-97816efa.csv'
```


parse labeled dataset

to do: include error message if api key not detected

```
nohup python3 data-ops/data-ops.py parse_dataset --dataset_path /output/data/labeled --output_path /output/data/labeled.csv &> /output/results/parsing_labeled.out &
```


# Dataset curation

## Fastdup

to do: add script

https://github.com/visual-layer/fastdup

https://visual-layer.readme.io/docs/analyzing-labeled-images

docker-compose exec machine_learning bash

jupyter notebook --port 5051 --ip 0.0.0.0 --no-browser --allow-root

## Cleanlab

to do: add script for analysis 

https://github.com/cleanlab/cleanlab

https://docs.cleanlab.ai/stable/tutorials/image.html



## Pixplot

to do: use GPU

docker-compose exec pixplot bash

Install pixplot as in https://github.com/YaleDHLab/pix-plot

to do: add arguments and metadata

pixplot --images "/output/results/iter_6/sample/*.jpg"

python -m http.server 5000




# Deployment as API

Flask






# Legacy


https://github.com/HumanSignal/label-studio/issues/3987

Create docker image

```
docker build . -t watermark_image
```

Run docker container

docker run --gpus all --env LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true --env LOCAL_FILES_SERVING_ENABLED=true --env LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/files --env LOCAL_FILES_DOCUMENT_ROOT=/label-studio/files -p 8092:8092 -v /home/jcejudo/projects/watermark_classification:/output -v /home/jcejudo/projects/watermark_classification:/label-studio/files -v $(pwd)/mydata:/label-studio/data -v $(pwd):/code -it watermark_image:latest

docker run --gpus all -p 8092:8092 -v /home/jcejudo/projects/watermark_classification:/output -v /home/jcejudo/projects/watermark_classification:/label-studio/files -v $(pwd)/mydata:/label-studio/data -v $(pwd):/code -it watermark_image:latest
























