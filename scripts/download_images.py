import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests
from PIL import Image
from io import BytesIO
import fire

def get_image(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # This will raise an HTTPError for bad responses
        img = Image.open(BytesIO(response.content))
        return img
    except requests.RequestException as e:
        print(f"Failed to download {url}: {e}")
        return None

def save_image(img, path):
    try:
        img.save(path)
        return path.name
    except IOError as e:
        print(f"Failed to save image {path.name}: {e}")
        return None

def download_images(df, saving_dir, n_images=10):
    watermark_dir = saving_dir.joinpath('watermark')
    watermark_dir.mkdir(parents=True, exist_ok=True)

    no_watermark_dir = saving_dir.joinpath('no_watermark')
    no_watermark_dir.mkdir(parents=True, exist_ok=True)

    df = df.sample(frac=1).reset_index(drop=True)[:n_images]  # Limit the number of images to `n_images`
    tasks = [(row['image_url'], saving_dir.joinpath(row['category'], row['europeana_id'].replace('/', '[ph]') + '.jpg')) for index, row in df.iterrows()]

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = []
        # Submit download tasks
        for url, path in tasks:
            future = executor.submit(get_image, url)
            results.append((future, path))

        # Save images as they are downloaded
        for future, path in tqdm(results, total=len(results), desc="Downloading Images"):
            img = future.result()
            if img:
                save_image(img, path)

def main(**kwargs):
    saving_dir = kwargs.get('saving_dir')
    input_path = kwargs.get('input')

    saving_dir = Path(saving_dir)
    saving_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    print(f'Downloading {df.shape[0]} images')
    download_images(df, saving_dir, n_images=int(1e6))  # Adjust `n_images` as needed
    print('Finished')

if __name__ == "__main__":
    fire.Fire(main)
