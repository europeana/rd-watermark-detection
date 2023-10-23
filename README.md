# Watermarks

Create docker image

`docker build . -t watermark_image`

Run docker container

`docker run --gpus all -p 8090:8090 -v /home/jcejudo/projects/watermark_classification:/output -v $(pwd):/code -it watermark_image:latest`

`docker run --gpus all -v /home/jcejudo/projects/watermark_classification:/output -v $(pwd):/code -it watermark_image:latest`




Data acquisition


`nohup python3 data-ops/harvest_data.py --n_per_dataset 10 --saving_path /output/data/testing.csv --labeled_path /output/data/parsed_dataset.csv &> /output/results/testing.out &`

`nohup python3 data-ops/download_images.py --input /output/data/testing.csv --saving_dir /output/data/testing &> /output/results/download_images.out &`


Model training

`nohup python3 machine-learning/train.py --data_dir /output/data/labeled --saving_dir /output/results/iter_testing --max_epochs 2 --sample 0.25 &> /output/results/training.out &`

`python3 machine-learning/evaluate.py --results_path /output/results/iter_5 --saving_path /output/results/iter_5`



Predict

`python3 machine-learning/predict.py --input /output/data/unlabeled --results_path /output/results/iter_5 --metadata /output/data/unlabeled.csv --saving_path /output/results/iter_5/predictions.csv --mode uncertain --n_predictions 100 --sample 0.1 --batch_size 64`

Label studio

label-studio -p 8090






conda activate water_env
pip install -r requirements.txt

pip install https://github.com/heartexlabs/label-studio/archive/master.zip

# Get noisy dataset

`nohup python data-ops/harvest_data.py --saving_path /home/jcejudo/projects/watermark_classification/data/unlabeled.csv --labeled_path /home/jcejudo/projects/watermark_classification/data/parsed_dataset.csv &> harvest_data.out &`

`nohup python data-ops/download_images.py --input /home/jcejudo/projects/watermark_classification/data/unlabeled.csv --saving_dir /home/jcejudo/projects/watermark_classification/data/unlabeled &> download_images.out &`



# Train first model

`nohup python machine-learning/train.py --data_dir /home/jcejudo/projects/watermark_classification/data/labeled --saving_dir /home/jcejudo/projects/watermark_classification/results/iter_6 --max_epochs 20 --sample 1.0 &> training.out &`

`python machine-learning/evaluate.py --results_path /home/jcejudo/projects/watermark_classification/results/iter_6 --saving_path /home/jcejudo/projects/watermark_classification/results/iter_6/`

# Human in the loop


`python machine-learning/predict.py --input /home/jcejudo/projects/watermark_classification/data/unlabeled --results_path /home/jcejudo/projects/watermark_classification/results/iter_5 --metadata /home/jcejudo/projects/watermark_classification/data/unlabeled.csv --saving_path /home/jcejudo/projects/watermark_classification/results/iter_5/predictions.csv --mode uncertain --n_predictions 750 --sample 1.0 --batch_size 64`

label-studio start

import predictions to labelstudio, label, export annotations


moving annotations to labeled and removing from unlabeled images

`python data-ops/move_labeled.py --unlabeled_dir /home/jcejudo/projects/watermark_classification/data/unlabeled --labeled_dir /home/jcejudo/projects/watermark_classification/data/labeled --labels '/home/jcejudo/projects/watermark_classification/results/iter_5/project-15-at-2023-10-09-18-59-31d43bb2.csv'`













