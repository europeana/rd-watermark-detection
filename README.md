# R&D watermark detection repository
## Table of Contents
* [Table of Contents](#table-of-contents)
* [About the Project](#about-the-project)
* [Description](#description)
* [Built With](#built-with)
* [Prerequisites](#prerequisites)
* [Installation](#installation-of-library)
* [Usage](#usage)
    * [Model building](#model-building)
        * [Data acquisition](#data-acquisition)
        * [Model training](#model-training)
        * [Inference](#inference)
        * [Hyperparameter tuning](#hyperparameter-tuning)
        * [Dataset curation](#dataset-curation)
    * [Data annotation](#data-annotation)

# About the Project

# Description

This repository contains a set of tools for building a model to detect digital watermarks in images, spanning data acquisition, labeling and curation, as well as model training and evaluation. 

# Built With
* [Python](https://www.python.org/)
* [PyEuropeana](https://github.com/europeana/rd-europeana-python-api/tree/master)
* [PyTorch Lightning](https://lightning.ai/pytorch-lightning)
* [LabelStudio](https://labelstud.io/)

# Prerequisites

* [Docker Engine](https://docs.docker.com/engine/)
* [Docker Compose](https://docs.docker.com/compose/)

# Installation

Get the repository:

```shell
git clone https://github.com/europeana/rd-watermark-detection.git
```

Go to the `.env` file in the cloned folder of the repo `/rd-watermark-detection` and add paths for the data, LabelStudio metadata and ports for jupyter notebooks, tensorboard and LabelStudio.

```
DATA_DIR="/path/to/data"
LABELSTUDIO_DIR="/path/for/labelstudio/metadata"
NOTEBOOK_PORT=5051
TENSORBOARD_PORT=6006
LABELSTUDIO_PORT=8093
```

Docker can be used to set up two containers:
* ```machine_learning```: for model building
* ```label_studio```: for data annotation using LabelStudio

Run the following command to build and run the containers: 

```shell
docker-compose up -d
```


# Usage

## Model building

Execute the following command to access the ```machine_learning``` container

```shell
docker-compose exec machine_learning bash
```

Jupyter notebook can be launched with the following command:

```shell
jupyter notebook --port 5051 --ip 0.0.0.0 --no-browser --allow-root
```

Make sure the port is the same as in the .env file


### Data acquisition

The first step is to obtain data for assembling a dataset of images with and without watermarks. We can use the Europeana Search API for this via PyEuropeana. Get your Europeana API key [here](https://pro.europeana.eu/page/get-api) and set it up as an environment variable as follows:

```shell
export EUROPEANA_API_KEY=yourapikey
```

The data sources for watermarked images can be specified with a json file with the following structure:

```
{"query_list": [
    "edm_datasetName:2024905_*", 
    "edm_datasetName:2059505_*", 
    "edm_datasetName:334_*",
     ]}

```

Non-watermarked images will be obtained from arbitrary collections using the query `"*"` in the Search API. 

Run the following command to obtain the metadata of the objects resulting from the aforementioned queries. The number of non-watermark images will be the same as the number of watermarked images. The number of images per dataset can be set with the parameter `n_per_dataset`:

```shell
nohup python3 scripts/data_ops.py harvest_data \
 --datasets_path /storage/data/new_datasets.json \
 --n_per_dataset 50 \
 --saving_path /storage/data/unlabeled.csv \
 --labeled_path /storage/data/labeled_4312.csv \
 &> /storage/results/data_harvesting.out &
```

Run the following command to download the image files of the objects retrieved in the previous step:

```
nohup python3 scripts/data_ops.py download \
 --input /storage/data/unlabeled.csv \
 --saving_dir /storage/data/unlabeled \
 &> /storage/results/download_images.out &

```

Before training a model you might want to manually review and annotate the data using labelstudio. Go to the section [Data annotation](#data-annotation)


### Model training

Once we have a labeled dataset we can train an image classification model that will separate images with and without watermarks. The model is a [ResNet 18](https://pytorch.org/hub/pytorch_vision_resnet/) pretrained on ImageNet with a binary output layer. For training the labeled dataset is divided in train/val/test splits with stratified sampling. The validation loss during training is monitored on the validation set, and the resulting model is evaluated on the test set. 

The following command trains a model as described above taking the values for the batch size, learning rate, max number of epochs, and crossvalidation as arguments:

```shell
nohup python3 scripts/machine_learning.py train \
 --batch_size 16 \
 --data_dir /storage/data/labeled_6999 \
 --saving_dir /storage/results/labeled_6999 \
 --max_epochs 40 \
 --sample 1.0 \
 --crossvalidation True \
 &> /storage/results/training.out &
```

The training can be monitored using tensorboard:

```shell
tensorboard --port 6006 --host 0.0.0.0 --logdir=/storage/results/labeled_6999/split_2/tensorboard_logs/
```

The results of the training will be a set of files with the model weights, the data splits and evalutation metrics. There are also interpretability maps using GradCAM, which has been adapted from [this repository](https://github.com/jacobgil/pytorch-grad-cam)


### Inference

Once the model has been trained it can be used to predict on unseen images

uncertain

Useful for Active Learning


```shell
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

### Hyperparameter tuning

```shell
nohup python3 scripts/hyperparameter_tuning.py \
 --data_dir /storage/data/labeled_4312 \
 --saving_dir /storage/results/hyperparameter_tuning \
 --num_epochs 15 \
 --num_samples 50 \
 &> /storage/results/hyperparameter_tuning.out &
```

References:
* [Ray tune with Pytorch Lightning](https://docs.ray.io/en/latest/tune/examples/tune-pytorch-lightning.html)
* [Example Pytorch Lightning MNIST](https://docs.ray.io/en/latest/train/examples/lightning/lightning_mnist_example.html#lightning-mnist-example)
* [Pytorch hyperparameter tuning](https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html)
* [Ray tune Docker images](https://docs.ray.io/en/latest/ray-overview/installation.html#docker-source-images)

to do: add path for temp files https://github.com/ray-project/ray/issues/31478


automatic image augmentation

https://albumentations.ai/docs/autoalbument/
https://albumentations.ai/docs/autoalbument/docker/

(to be added)


### Dataset curation

#### Fastdup

https://github.com/visual-layer/fastdup

https://visual-layer.readme.io/docs/analyzing-labeled-images

```
python3 scripts/dataset_curation.py fastdup \
 --data_dir /storage/data/labeled_4312 \
 --saving_dir /storage/results/fastdup
```

Explore the html pages in saving_dir


#### Cleanlab

https://github.com/cleanlab/cleanlab

https://docs.cleanlab.ai/stable/tutorials/image.html

First run crossvalidation using the machine_learning.py script with crossvalidation=True

```
python3 scripts/dataset_curation.py cleanlab --results_dir /storage/results/iter_7/ 
```



## Data Annotation

label_studio

```shell
docker-compose exec label_studio bash
```

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






