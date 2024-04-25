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
 &> /storage/logs/data_harvesting.out &
```

Run the following command to download the image files of the objects retrieved in the previous step:

```
nohup python3 -u scripts/data_ops.py download \
 --input /storage/data/labeled_from_record.csv \
 --saving_dir /storage/data/labeled_from_record \
 &> /storage/logs/download_images.out &

```

Before training a model you might want to manually review and annotate the data using labelstudio. Go to the section [Data annotation](#data-annotation)


```
nohup python3 scripts/get_collections_data.py \
 --saving_path /storage/data/collections.csv \
 &> /storage/logs/get_collections_data.out &
```


```
nohup python3 -u scripts/get_labeled_dataset.py \
 --input_path /storage/data/watermark_record.csv \
 --output_path /storage/data/labeled_from_record.csv \
 &> /storage/logs/get_labeled_dataset.out &
```

```
nohup python3 -u scripts/data_ops.py download \
 --input /storage/data/labeled_from_record.csv \
 --saving_dir /storage/data/labeled_from_record \
 &> /storage/logs/download_images.out &

```

In rd-img-utilities

nohup python3 -u scripts/download_images.py --labeled True --sample 1.0 --suffix 'LARGE' --input /storage/data/labeled_from_record.csv --output /storage/data/labeled_from_record &> /storage/logs/download_images.out &


### Model training

Once we have a labeled dataset we can train an image classification model that will separate images with and without watermarks. The model is a [ResNet 18](https://pytorch.org/hub/pytorch_vision_resnet/) pretrained on ImageNet with a binary output layer. For training the labeled dataset is divided in train/val/test splits with stratified sampling. The validation loss during training is monitored on the validation set, and the resulting model is evaluated on the test set. 

The following command trains a model as described above taking the values for the batch size, learning rate, max number of epochs, and crossvalidation as arguments:

```shell
nohup python3 -u scripts/machine_learning.py train \
 --batch_size 64 \
 --learning_rate 5e-5 \
 --model_size 18 \
 --data_dir /storage/data/labeled_55255/ \
 --saving_dir /storage/results/labeled_55255/ \
 --max_epochs 50 \
 --sample 1.0 \
 --crossvalidation False \
 &> /storage/logs/training.out &
```

The training can be monitored using tensorboard:

```shell
tensorboard --port 6006 --host 0.0.0.0 --logdir=/storage/results/labeled_55255/split_1/tensorboard_logs/
```

The results of the training will be a set of files with the model weights, the data splits and evalutation metrics. There are also interpretability maps using GradCAM, which has been adapted from [this repository](https://github.com/jacobgil/pytorch-grad-cam)


### Inference

Once the model has been trained it can be used to predict on unseen images. The resulting predictions can be ranked by their uncertainty, defined in this case as the absolute value of the difference between the confidence scores for the two output labels. This can be useful for sampling data to be annotated using an Active Learning or human-in-the-loop approach. 

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
 &> /storage/logs/predict.out &
```

Deployment

```shell
nohup python3 -u deployment/inference.py \
 --input /storage/data/inference \
 --results_path /storage/results/labeled_23555/split_1 \
 --metadata /storage/data/sample_inference.csv \
 --saving_path /storage/results/results_inference.csv \
 --sample 1.0 \
 --batch_size 32 \
 &> /storage/logs/inference.out &
```

### Hyperparameter tuning

Deep Learning models can be very sensitive to the value of certain hyperparameters. With the following command a hyperparameter search is carried for the batch size and learning rate. The range of the possible values of these parameters can be adjusted in the script

```shell
nohup python3 -u scripts/hyperparameter_tuning.py \
 --data_dir /storage/data/labeled_23555 \
 --saving_dir /storage/results/hyperparameter_tuning \
 --num_epochs 10 \
 --num_samples 20 \
 --grace_period 2 \
 --sample 0.5 \
 &> /storage/logs/hyperparameter_tuning.out &
```

References:
* [Ray tune with Pytorch Lightning](https://docs.ray.io/en/latest/tune/examples/tune-pytorch-lightning.html)
* [Example Pytorch Lightning MNIST](https://docs.ray.io/en/latest/train/examples/lightning/lightning_mnist_example.html#lightning-mnist-example)
* [Pytorch hyperparameter tuning](https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html)
* [Ray tune Docker images](https://docs.ray.io/en/latest/ray-overview/installation.html#docker-source-images)

to do: add path for temp files https://github.com/ray-project/ray/issues/31478


Automatic Image Augmentation

(to be added maybe)

References:
* [Autoalbument](https://albumentations.ai/docs/autoalbument/)
* [Autoalbument Docker](https://albumentations.ai/docs/autoalbument/docker/)


### Dataset curation

When working with thousands of images it is common to have noisy labels, misclassified images, duplicates and outliers. These can harm the performance of a model when training so it is good to identify them. 

#### Fastdup

Fastdup creates a report containing duplicate images, outliers and clusters. 

```
python3 scripts/dataset_curation.py fastdup \
 --data_dir /storage/data/labeled_4312 \
 --saving_dir /storage/results/fastdup
```

After running the previous command explore the html pages in saving_dir

References:
* [Fastdup repository](https://github.com/visual-layer/fastdup)
* [Analizing image classification dataset](https://visual-layer.readme.io/docs/analyzing-labeled-images)



#### Cleanlab

Cleanlab helps identifying mislabeled images. First run crossvalidation using the `machine_learning.py` script with `crossvalidation=True`.

```
python3 scripts/dataset_curation.py cleanlab --results_dir /storage/results/iter_7/ 
```

References:
* [Cleanlab repository](https://github.com/cleanlab/cleanlab)
* [Image classification tutorial](https://docs.cleanlab.ai/stable/tutorials/image.html)



## Data Annotation

Access the `label_studio` container:

```shell
docker-compose exec label_studio bash
```

Start watermark_detection project

```shell
label-studio init watermark_detection --label-config /code/labelstudio-config.xml

label-studio start watermark_detection -p 8093 --host localhost
```

Go to the watermark_detection project -> Settings -> Cloud Storage -> Source storage -> Local storage and add the path to the sample to annotate. Annotate, export as CSV and place it in the data folder. 

Execute the following command to correct the annotations of the labeled dataset:

```shell
python3 scripts/data_ops.py move_labeled \
 --sample_dir /storage/results/iter_6/sample_certain \
 --labeled_dir /storage/data/labeled_new_data \
 --labels '/storage/results/iter_6/project-1-at-2023-10-31-14-05-97816efa.csv'
```
Run the following command to parse the corrected labeled dataset:

```shell
nohup python3 scripts/data_ops.py parse_dataset \
 --dataset_path /storage/data/labeled_6999 \
 --output_path /storage/data/labeled_6999.csv \
 &> /storage/results/parsing_labeled.out &
```

References:
* [LabelStudio docker compose](https://github.com/HumanSignal/label-studio#run-with-docker-compose)
* [LabelStudio issue](https://github.com/HumanSignal/label-studio/issues/3242)









