# Detection of watermarks in images

```
docker-compose up -d
```

Services:
machine_learning

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


## Data acquisition

```
nohup python3 scripts/data-ops.py harvest_data \
 --datasets_path /storage/data/new_datasets.json \
 --n_per_dataset 200 \
 --saving_path /storage/data/unlabeled.csv \
 --labeled_path /storage/data/labeled_4312.csv \
 &> /storage/results/data_harvesting.out &

nohup python3 scripts/data-ops.py download \
 --input /storage/data/unlabeled.csv \
 --saving_dir /storage/data/unlabeled \
 &> /storage/results/download_images.out &

```


## Model training

to do:
tensorboard
hyperparameter tuning
automatic image augmentation
XAI

https://lightning.ai/docs/pytorch/stable/extensions/logging.html
https://github.com/jacobgil/pytorch-grad-cam
https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
https://albumentations.ai/docs/autoalbument/

```
nohup python3 scripts/machine-learning.py train \
 --batch_size 16 \
 --data_dir /storage/data/labeled_4312 \
 --saving_dir /storage/results/iter_7 \
 --max_epochs 2 \
 --sample 0.1 \
 --crossvalidation False \
 &> /storage/results/training.out &
```

## Predict

```
python3 scripts/machine-learning.py predict \
 --input /storage/data/unlabeled \
 --results_path /storage/results/iter_6 \
 --metadata /storage/data/unlabeled.csv \
 --saving_path /storage/results/iter_6/predictions.csv \
 --mode uncertain \
 --n_predictions 800 \
 --sample 1.0 \
 --batch_size 64 \
 --sample_path /storage/results/iter_6/sample
```

## Annotate with Label-Studio

Access the container of label studio

Start watermark_detection project

```
label-studio init watermark_detection --label-config /code/labelstudio-config.xml

label-studio start watermark_detection -p 8093
```

Go to the watermark_detection project -> Settings -> Cloud Storage -> Source storage -> Local storage

Annotate and export as CSV

moving annotations to labeled and removing from unlabeled images

```
python3 scripts/data-ops.py move_labeled \
 --sample_dir /storage/results/iter_6/sample \
 --labeled_dir /storage/data/labeled \
 --labels '/storage/results/iter_6/project-1-at-2023-10-31-14-05-97816efa.csv'
```
parse labeled dataset

to do: include error message if api key not detected

```
nohup python3 scripts/data-ops.py parse_dataset \
 --dataset_path /storage/data/labeled \
 --output_path /storage/data/labeled.csv \
 &> /storage/results/parsing_labeled.out &
```


# Dataset curation

## Fastdup

https://github.com/visual-layer/fastdup

https://visual-layer.readme.io/docs/analyzing-labeled-images

```
python3 scripts/dataset-curation.py fastdup \
 --data_dir /storage/data/labeled_4312 \
 --saving_dir /storage/results/fastdup
```

Explore the html pages in saving_dir


## Cleanlab

to do: add script for analysis 

https://github.com/cleanlab/cleanlab

https://docs.cleanlab.ai/stable/tutorials/image.html

First run crossvalidation using the machine_learning.py script with crossvalidation=True

```
python3 scripts/dataset-curation.py cleanlab --results_dir /storage/results/iter_6/ 
```



