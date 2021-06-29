import yaml
import collections
import torch

import sys
sys.path.append('..')
from dataset import SportsDataset
from utils.transforms import train_transform, valid_transform
from utils.models import get_model, get_device
from glob import glob

import torch.nn as nn
from catalyst.runners.runner import SupervisedRunner
from catalyst.callbacks import EarlyStoppingCallback
from catalyst.contrib.nn import OneCycleLRWithWarmup
from catalyst.callbacks.metrics.classification import MultilabelPrecisionRecallF1SupportCallback

with open(r'../configs/config.yaml') as file:
    conf = yaml.load(file, Loader=yaml.FullLoader)    


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
        
    valid_tr = valid_transform(conf['dataset']['height'], conf['dataset']['width'])
    train_tr = train_transform(conf['dataset']['height'], conf['dataset']['width'])

    train_ds = SportsDataset(
        list_of_images=glob(f"{conf['dataset']['list_of_images']}/*/*.jpg"),
        labels=conf['dataset']['labels'],
        transform=train_tr,
        seed=conf['dataset']['seed'],
    )

    print('Train dataset loaded')

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=conf['dataset']['batch_size'],
        shuffle=True, num_workers=conf['dataset']['num_workers']
    )

    val_ds = SportsDataset(
        list_of_images=glob(f"{conf['dataset']['list_of_images']}/*/*.jpg"),
        labels=conf['dataset']['labels'],
        transform=train_tr,
        is_train=False,
        seed=conf['dataset']['seed'],
    )

    print('Valid dataset loaded')

    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=conf['dataset']['batch_size'],
        shuffle=True, num_workers=conf['dataset']['num_workers'])

    loaders = collections.OrderedDict()
    loaders['train'] = train_loader
    loaders['valid'] = val_loader

    device = get_device()
    model = get_model(
        arch=conf['model']['arch'],
        model_name=conf['model'].get('encoder'),
        num_classes=len(conf['dataset']['labels']),
        pretrained=conf['model'].get('pretrained')
    ).to(device)

    print('molde created')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    runner = SupervisedRunner(
        input_key="features", output_key="scores",
        target_key="targets", loss_key="loss",
    )

    print('Start traning')

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir=conf['main']['logdir'],
        num_epochs=conf['main']['epoches'],
        callbacks=[
            EarlyStoppingCallback(
                patience=conf['main']['patience'],
                loader_key='valid',
                min_delta=conf['main']['min_delta'],
                metric_key='loss',
                minimize=True,
            ),
        ],
        verbose=True,
)
