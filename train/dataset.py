import yaml
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as alb
from sklearn.model_selection import train_test_split

import sys
sys.path.append('..')
from utils.models import get_device

from typing import List, Union


class SportsDataset(Dataset):

    def __init__(
        self, 
        list_of_images: List[str],
        labels: List[str],
        transform: alb.core.composition.Compose = None,
        is_train: bool = True,
        val_size: float = 0.15,
        device: str = None,
        seed: int = 42,
    ):
        self.labels = {k: v for k, v in zip(labels, np.arange(len(labels)))}
        self.is_train = is_train
        self.val_size = val_size
        self._split_data(list_of_images, seed)
        self.device = device if device else get_device()
        self.transform = transform
    
    def _split_data(self, list_of_images: List[str], seed: int):
        self.images = []
        if self.val_size:
            for label in self.labels:
                label_images = [x for x in list_of_images if label == x.split('/')[-2]]
                X_train, X_test = train_test_split(label_images, test_size=self.val_size, random_state=seed)
                images = X_train if self.is_train else X_test
                self.images.extend(images)
            
    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> List[Union[torch.Tensor, int]]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        path_to_image = self.images[idx]
        image = cv2.imread(path_to_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[path_to_image.split('/')[-2]]
        
        if self.transform:
            image = self.transform(image=image)['image']
        image = image.to(self.device)
        return [image, label]
