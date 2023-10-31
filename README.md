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

```
nohup python3 scripts/machine-learning.py train \
 --batch_size 16 \
 --data_dir /storage/data/labeled_4312 \
 --saving_dir /storage/results/iter_6 \
 --max_epochs 1 \
 --sample 0.1 \
 --crossvalidation True \
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
python3 scripts/dataset-curation.py fastdup --data_dir /storage/data/labeled_4312 --saving_dir /storage/results/fastdup
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




## Pixplot

to do: use GPU

```
docker-compose exec pixplot bash
```

Install pixplot as in https://github.com/YaleDHLab/pix-plot

to do: add arguments and metadata

```
pixplot --images "/storage/results/iter_6/sample/*.jpg"

python -m http.server 5000
```

# Deployment as API

Flask

to do:
add script
add port
add Flask in requirements






# Legacy


https://github.com/HumanSignal/label-studio/issues/3987

Create docker image

```
docker build . -t watermark_image
```

Run docker container

docker run --gpus all --env LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true --env LOCAL_FILES_SERVING_ENABLED=true --env LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/files --env LOCAL_FILES_DOCUMENT_ROOT=/label-studio/files -p 8092:8092 -v /home/jcejudo/projects/watermark_classification:/output -v /home/jcejudo/projects/watermark_classification:/label-studio/files -v $(pwd)/mydata:/label-studio/data -v $(pwd):/code -it watermark_image:latest

docker run --gpus all -p 8092:8092 -v /home/jcejudo/projects/watermark_classification:/output -v /home/jcejudo/projects/watermark_classification:/label-studio/files -v $(pwd)/mydata:/label-studio/data -v $(pwd):/code -it watermark_image:latest
























