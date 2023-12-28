# R&D watermark detection repository
## Table of Contents
- [Table of Contents](#table-of-contents)
- [About the Project](#about-the-project)
- [Description](#description)
- [Built With](#built-with)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites-and-dependencies)
- [Installation](#installation-of-library)
- [Usage](#usage)

# About the Project

# Description

This repository contains a set of tools for building a model to detect digital watermarks in images, spanning data acquisition, labeling and curation, as well as model training and evaluation. 

# Built With
* [Python](https://www.python.org/)
* [PyEuropeana](https://github.com/europeana/rd-europeana-python-api/tree/master)
* [PyTorch Lightning](https://lightning.ai/pytorch-lightning)
* [LabelStudio](https://labelstud.io/)

# Getting Started
# Prerequisites and Dependencies

* [Docker Engine](https://docs.docker.com/engine/)
* [Docker Compose](https://docs.docker.com/compose/)

# Installation of library
### Get the repository
```shell
git clone https://github.com/europeana/rd-watermark-detection.git
```

### Build docker images

Go to the cloned folder of the repo /rd-watermark-detection,

Docker containers for machine learning and for deploying LabelStudio


and run the following command to build the containers
```shell
docker-compose build --no-cache
```

# Usage

## Enable Python API Endpoint for image enhancement
go to the cloned folder of the library /metis-image-enhancer/metis-image-enhancer-python-rest/src/main, 
and run the following commands to get the service up and running 
this command will run with 1 node
```shell
docker compose up 
```
the application uses a python api to enhance the image


## Setting up environment

.env file with environment variables: ports and paths

```
DATA_DIR="/path/to/data"
LABELSTUDIO_DIR="/path/for/labelstudio/metadata"
```

```shell
docker-compose up -d
```

Services:
* ```machine_learning```
* ```label_studio```

```
docker-compose exec machine_learning bash
```

label_studio

```
docker-compose exec label_studio bash
```

```
jupyter notebook --port 5051 --ip 0.0.0.0 --no-browser --allow-root
```


#### Data acquisition

Obtain data

Equal amount of images with and without watermarks

Set up your Europeana API key as

```
export EUROPEANA_API_KEY=yourapikey
```


```
nohup python3 scripts/data_ops.py harvest_data \
 --datasets_path /storage/data/new_datasets.json \
 --n_per_dataset 50 \
 --saving_path /storage/data/unlabeled.csv \
 --labeled_path /storage/data/labeled_4312.csv \
 &> /storage/results/data_harvesting.out &

nohup python3 scripts/data_ops.py download \
 --input /storage/data/unlabeled.csv \
 --saving_dir /storage/data/unlabeled \
 &> /storage/results/download_images.out &

```


## Model training

```
nohup python3 scripts/machine_learning.py train \
 --batch_size 16 \
 --data_dir /storage/data/labeled_6999 \
 --saving_dir /storage/results/labeled_6999 \
 --max_epochs 40 \
 --sample 1.0 \
 --crossvalidation True \
 &> /storage/results/training.out &
```

```
tensorboard --port 6006 --host 0.0.0.0 --logdir=/storage/results/labeled_6999/split_2/tensorboard_logs/
```

https://github.com/jacobgil/pytorch-grad-cam



## Predict


```
nohup python3 scripts/machine_learning.py predict \
 --input /storage/data/unlabeled \
 --results_path /storage/results/iter_6/split_5 \
 --metadata /storage/data/unlabeled.csv \
 --saving_path /storage/results/iter_6/split_5/predictions.csv \
 --mode uncertain \
 --n_predictions 1000 \
 --sample 1.0 \
 --batch_size 16 \
 --sample_path /storage/results/iter_6/sample_certain \
 &> /storage/results/predict.out &
```

## Annotate with Label-Studio

Run it with docker compose 

https://github.com/HumanSignal/label-studio#run-with-docker-compose


https://github.com/HumanSignal/label-studio/issues/3242

Access the container of label studio

```
docker-compose exec label_studio bash
```

Start watermark_detection project

```
label-studio init new_project --label-config /code/labelstudio-config.xml

label-studio start new_project -p 8093 --host localhost
```

Go to the watermark_detection project -> Settings -> Cloud Storage -> Source storage -> Local storage

Add the path to the sample to annotate

Annotate and export as CSV. Move to some part of the results folder

moving annotations to labeled and removing from unlabeled images

```
python3 scripts/data_ops.py move_labeled \
 --sample_dir /storage/results/iter_6/sample_certain \
 --labeled_dir /storage/data/labeled_new_data \
 --labels '/storage/results/iter_6/project-1-at-2023-10-31-14-05-97816efa.csv'
```
parse labeled dataset

to do: include error message if api key not detected

```
nohup python3 scripts/data_ops.py parse_dataset \
 --dataset_path /storage/data/labeled_6999 \
 --output_path /storage/data/labeled_6999.csv \
 &> /storage/results/parsing_labeled.out &
```

## Hyperparameter tuning

to do: add path for temp files
https://github.com/ray-project/ray/issues/31478

hyperparameter tuning

https://docs.ray.io/en/latest/tune/examples/tune-pytorch-lightning.html
https://docs.ray.io/en/latest/train/examples/lightning/lightning_mnist_example.html#lightning-mnist-example

https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
https://docs.ray.io/en/latest/ray-overview/installation.html#docker-source-images


```
nohup python3 scripts/hyperparameter_tuning.py \
 --data_dir /storage/data/labeled_4312 \
 --saving_dir /storage/results/hyperparameter_tuning \
 --num_epochs 15 \
 --num_samples 50 \
 &> /storage/results/hyperparameter_tuning.out &
```


automatic image augmentation

https://albumentations.ai/docs/autoalbument/
https://albumentations.ai/docs/autoalbument/docker/




# Dataset curation

## Fastdup

https://github.com/visual-layer/fastdup

https://visual-layer.readme.io/docs/analyzing-labeled-images

```
python3 scripts/dataset_curation.py fastdup \
 --data_dir /storage/data/labeled_4312 \
 --saving_dir /storage/results/fastdup
```

Explore the html pages in saving_dir


## Cleanlab

https://github.com/cleanlab/cleanlab

https://docs.cleanlab.ai/stable/tutorials/image.html

First run crossvalidation using the machine_learning.py script with crossvalidation=True

```
python3 scripts/dataset_curation.py cleanlab --results_dir /storage/results/iter_7/ 
```



