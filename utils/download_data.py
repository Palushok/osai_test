import contextlib
import os
import matplotlib.pyplot as plt

from glob import glob
from pathlib import Path
from typing import List
from tqdm import tqdm
from urllib import request



def download_image(url: str, out_path: str) -> None:
    """
    'url': url for downloading
    'out_path': path to output file
    """
    try:
        req = request.Request(url, headers={'User-Agent': 'Mozilla/5.0','Content-Type': 'image/jpeg'})
        with contextlib.closing(request.urlopen(req, timeout=1)) as content:
            with open(out_path, "wb") as f:
                f.write(content.read())
    except:
        pass


def download_dataset(file: str, path_to_output: str, dataset: str) -> None:
    """
    'file': path to file with links
    'path_to_output': path to new folder for all data 
    'dataset': name of dataset ('train' or 'test')
    """
    
    label = Path(file).stem
    print(f'Download {label} in {dataset}')
    
    label_folder = Path(path_to_output) / dataset / label
    label_folder.mkdir(parents=True, exist_ok=True)
    print(label_folder)
    with open(file, 'r') as f:
        links = f.read().splitlines()
    for i, url in tqdm(enumerate(links), total=len(links)):
        output_file = label_folder / f'{i}.jpg'
        download_image(url, output_file)

def remove_trash_images(all_images: List[str]) -> None:
    """
    'all_images': list of paths to images
    """
    for image in all_images:
        try:
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            print(f'{image} deleted')
            os.remove(image)

if __name__ == '__main__':
    data = {
        'train': 'data/train/',
        'test': 'data/test/',
    }
    folder_for_images = 'data/images/'

    for split, path in data.items():
        for label in glob(path + '*'):
            download_dataset(label, folder_for_images, split)

    all_images = glob(f'{folder_for_images}/*/*/*.jpg')
    remove_trash_images(all_images)
