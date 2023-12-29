import os
import time
import ibm_boto3
from ibm_botocore.client import Config
from pathlib import Path

from machine_learning import *

"""

pip install ibm-cos-sdk

python3 scripts/performance_test.py download_images --n 1000 --saving_path /storage/data/thumnails

"""

def download_images(**kwargs):
    n = kwargs.get('n',100)
    saving_path = kwargs.get('saving_path')

    COS_API_KEY_ID = os.environ['COS_API_KEY_ID']
    COS_INSTANCE_CRN = os.environ['COS_INSTANCE_CRN']
    COS_ENDPOINT = os.environ['COS_ENDPOINT']

    # Create client
    client = ibm_boto3.resource("s3",
        ibm_api_key_id=COS_API_KEY_ID,
        ibm_service_instance_id=COS_INSTANCE_CRN,
        config=Config(signature_version="oauth"),
        endpoint_url=COS_ENDPOINT
    )

    start = time.time()

    count = 0

    print("Downloading...")

    bucket = client.Bucket('europeana-thumbnails-production')
    #object_summary_iterator = bucket.objects.limit(count=n)
    object_summary_iterator = bucket.objects.all()
    for item in object_summary_iterator:
        object = client.Object("europeana-thumbnails-production", item.key)
        if object.key.endswith('MEDIUM'):
            object.download_file(Path(saving_path).joinpath(f'{object.key}.jpg'))
            count += 1
            if count >= n:
                break

    end = time.time()

    dt = (end-start)/60.0

    print(f"Finished downloading {count} images, it took {dt:.2f} minutes")


def inference(**kwargs):

    results_path = kwargs.get('results_path')
    saving_path = kwargs.get('saving_path')
    input = kwargs.get('input')
    threshold = kwargs.get('threshold',0.5)

    sample = kwargs.get('sample',1.0)
    batch_size = kwargs.get('batch_size',4)
    metadata = kwargs.get('metadata')
    n_predictions = kwargs.get('n_predictions',200)

    mode = kwargs.get('mode','uncertain')
    sample_path = kwargs.get('sample_path')

    meta_df = pd.read_csv(metadata)

    results_path = Path(results_path)
    input = Path(input)

    with open(results_path.joinpath('classes.json'),'r') as f:
        classes = json.load(f)['classes']

    n_classes = len(classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Classifier(
        output_dim = n_classes, 
        learning_rate = 0.0,
        threshold = threshold
    )

    model.load_state_dict(torch.load(results_path.joinpath('checkpoint.pth')))

    print(device)

    model = model.to(device)

    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image_arr = np.array([str(path) for path in input.rglob('*.jpg')])
    # filter images in meta df
    id_arr = [fpath2id(path) for path in image_arr]
    idx = [True if id in meta_df['europeana_id'].values else False for id in id_arr]
    image_arr = image_arr[idx]
    # sample images for predicting
    idx = np.random.randint(image_arr.shape[0],size = int(sample*image_arr.shape[0]))
    image_arr = image_arr[idx]

    print(f'predicting on {image_arr.shape[0]} images')

    train_dataset = InferenceDataset(image_arr,transform = transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,collate_fn=my_collate)

    conf_list = []
    path_list = []
    prediction_list = []
    with torch.no_grad():
        for paths,batch in tqdm(train_loader):
            batch = batch.to(device)
            outputs = model(batch).cpu()
            prediction_list += [classes[i] for i in torch.argmax(outputs,axis=1)]
            conf_list.append(outputs)
            path_list.append(paths)

    print('Finished predicting')

    output = torch.Tensor(len(train_loader)*batch_size, n_classes).cpu()
    output = torch.cat(conf_list, out=output)

    path_list = [list(t) for t in path_list]
    path_list = [item for sublist in path_list for item in sublist]
    
    conf_dict = {'path':path_list}
    conf_dict.update({cat:output[:,i] for i,cat in enumerate(classes)})

    def absdiff(x):
        return np.abs(x[1]-x[2])

    df = pd.DataFrame(conf_dict)
    df['prediction'] = prediction_list
    df['absdiff'] = df.apply(absdiff, axis=1)

    if mode == 'uncertain': # for getting the most uncertain results
        df = df.sort_values(by=['absdiff'])
    else:
        df = df.sort_values(by=['absdiff'],ascending = False)

    df['europeana_id'] = df['path'].apply(fpath2id)

    df = df.merge(meta_df)
    df = df.drop_duplicates(subset=['path'])
    #df = df.head(n_predictions) 
    df.to_csv(saving_path,index=False)
    print(df.shape)

    # saving sample

    if sample_path:
        sample_path = Path(sample_path)
        sample_path.mkdir(parents = True, exist_ok = True)

        for path in df['path'].head(n_predictions).values:
            copyfile(path, sample_path.joinpath(Path(path).name))

    print('Finished')

def main(*args,**kwargs):
    arg = args[0]
    if arg == 'download_images':
        download_images(**kwargs)
    elif arg == 'inference':
        inference(**kwargs)


if __name__ == '__main__':
    fire.Fire(main)