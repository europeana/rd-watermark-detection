# Watermarks

conda activate water_env
pip install -r requirements.txt

pip install https://github.com/heartexlabs/label-studio/archive/master.zip

# Get noisy dataset

nohup python data-ops/harvest_data.py --saving_path /home/jcejudo/projects/watermark_classification/data/watermarks.csv &> harvest_data.out &

nohup python data-ops/download_images.py --input /home/jcejudo/projects/watermark_classification/data/watermarks.csv --saving_dir /home/jcejudo/projects/watermark_classification/data/unlabeled &> download_images.out &

to do: remove duplicates

# Train first model

nohup python machine-learning/train.py --data_dir /home/jcejudo/projects/watermark_classification/data/labeled --saving_dir /home/jcejudo/projects/watermark_classification/results/iter_0 --max_epochs 1 --sample 1.0 &> training.out &

python machine-learning/evaluate.py --results_path /home/jcejudo/projects/watermark_classification/results/iter_0 --saving_path /home/jcejudo/projects/watermark_classification/results/iter_0/

# Human in the loop


python machine-learning/predict.py --input /home/jcejudo/projects/watermark_classification/data/unlabeled --results_path /home/jcejudo/projects/watermark_classification/results/iter_0 --metadata /home/jcejudo/projects/watermark_classification/data/watermarks.csv --saving_path /home/jcejudo/projects/watermark_classification/results/iter_0/predictions.csv --mode uncertain --n_predictions 400 --sample 1.0

label-studio start

import predictions to labelstudio, label, export annotations


moving annotations to labeled and removing from unlabeled images

python data-ops/move_labeled.py --unlabeled_dir /home/jcejudo/projects/watermark_classification/data/unlabeled --labeled_dir /home/jcejudo/projects/watermark_classification/data/labeled --labels '/home/jcejudo/projects/watermark_classification/results/iter_0/project-14-at-2022-03-09-22-06-c16e6562.csv'













