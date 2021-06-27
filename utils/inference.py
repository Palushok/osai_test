import argparse
import cv2
import torch
import yaml
import numpy as np
import albumentations as alb

from glob import glob
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from typing import List
from numpy.random import choice

from models import get_model, get_device
from transforms import valid_transform


def predict(model, image_path: str, transforms: alb.Compose, labels: List[str], device: str):
    """
    model: torch model
    image_path: path to single image
    transforms: transforms used for inference
    labels: list of labels in right order
    device: 'cpu' or 'cuda' or '0', '1'...
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, 4)
    inp = vt(image=image)['image'].unsqueeze(0).to(device)
    pred_label = int(model.forward(inp).argmax().detach().cpu().numpy())
    return pred_label

def predict_folder(images, model, transfroms, labels, device, trues):
    """
    images: path to labeled data folder
    model: torch model
    transforms: transforms used for inference
    labels: list of labels in right order
    """
    preds = []

    for image_path in tqdm(images):
        pred = predict(model, image_path, transfroms, labels, device)
        preds.append(pred)

    f1 = f1_score(trues, preds, average='macro')
    acc = accuracy_score(trues, preds)
    return f1, acc

def get_trues(images, labels):
    trues = []
    for image_path in images:
        true = labels.index(image_path.split('/')[-2])
        trues.append(true)
    return trues

def random_predict(images_folder, labels, true):
    count_of_images = []
    for label in labels:
        count_of_images.append(len(glob(f'{images_folder}/{label}/*')))

    count_of_images = np.array(count_of_images)
    total_images = sum(count_of_images)
    parted_count = count_of_images/total_images

    random_preds = choice(
        np.arange(len(labels)), total_images, p=parted_count)
    
    f1 = f1_score(trues, random_preds, average='macro')
    acc = accuracy_score(trues, random_preds)

    return f1, acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', type=str, help='Path to config file', default='configs/config.yaml')
    parser.add_argument('-f', '--folder', type=str, help='Path to labeled data', default='data/images/test/')
    parser.add_argument('-w', '--weights', type=str, help='Path to model weights', default='weights/mobilenet_v3_small.pth')
    
    args = parser.parse_args()

    config_path = args.config
    images_folder = args.folder
    path_to_weights = args.weights

    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        
    labels = config['dataset']['labels']

    device = get_device()

    model = get_model(
        arch=config['model']['arch'],
        model_name=config['model'].get('encoder'),
        num_classes=len(config['dataset']['labels']),
        pretrained=config['model'].get('pretrained'),
    ).to(device)
    state_dict = torch.load(path_to_weights)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval();

    vt = valid_transform(config['dataset']['height'], config['dataset']['width'])

    images = glob(f'{images_folder}/*/*.jpg')
    trues = get_trues(images, labels)

    f1, acc = predict_folder(images, model, vt, labels, device, trues)
    f1_rand, acc_rand = random_predict(images_folder, labels, trues)

    print(f'F1 = {f1}\nAccuracy = {acc}')
    print('*********')
    print(f'Random weighted:\nF1 = {f1_rand}\nAccuracy = {acc_rand}')




