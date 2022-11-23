import torch
import torchvision
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from torchvision.datasets import ImageFolder

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from math import floor, ceil
from sklearn.model_selection import train_test_split

import shutil
import requests
import functools
import pathlib
from pathlib import Path
import shutil
from tqdm import tqdm
import os
from collections import defaultdict

matplotlib.rcParams['figure.figsize'] = (20, 5)

class MyDataset(Dataset):
    def __init__(self, path_x, data_y=None, transform=None):
        self.path_x = path_x
        self.files_x = sorted(os.listdir(path_x))
        self.data_y = data_y
        self.transform = transform

    def __len__(self):
        return len(self.files_x)

    def __getitem__(self, index):
        name = self.files_x[index]
        if self.data_y is not None:
            score = self.data_y[name]
        else:
            score = None
        image = plt.imread(self.path_x + '/' + name)
        if image.ndim != 3:
            image = np.dstack([image] * 3)

        if self.transform is not None:
            image = self.transform(image=image)['image']

        return image, 0, name

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


transform_train = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.3),
    A.Rotate(p=0.35, limit=15),
    A.RingingOvershoot(p=0.2, blur_limit=(3, 7)),
    A.OneOf([
        A.HueSaturationValue(p=0.3),
        A.RGBShift(p=0.3),
        A.Compose([ 
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.5),
            A.CLAHE(p=0.5),
        ], p=1)
    ], p=0.5),
    A.Affine(scale=(0.85, 1), translate_percent=(0, 0.10), shear=(-4, 4), p=0.35),
    A.Normalize(MEAN, STD),
    ToTensorV2(),
])

class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()

        features = torchvision.models.efficientnet_v2_s()
        features = list(features.children())[:-2]
        features = nn.Sequential(*features)

        self.features = features
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.8),
            nn.Linear(1280 , 50)
        )

        self.loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.acc = lambda pred, y: torch.sum(F.softmax(pred, dim=1).argmax(dim=1) == y) / y.shape[0]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        """the full training loop"""
        x, y = batch[0], batch[1]

        pred = self(x)
        loss = self.loss(pred, y)
        
        acc = self.acc(pred, y)

        return {'loss': loss, 'acc': acc}

    def configure_optimizers(self):
        """ Define optimizers and LR schedulers. """
        optimizer = torch.optim.Adam([
            {'params': self.features.parameters(), 'lr': 3e-5},
            {'params': self.classifier.parameters()}
        ], lr=3e-4, weight_decay=5e-4)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=0.2, 
            patience=5, 
            verbose=True)
            
        lr_dict = {
            # REQUIRED: The scheduler instance
            "scheduler": lr_scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "epoch",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": "val_acc"
        } 

        return [optimizer], [lr_dict]

    # OPTIONAL
    def validation_step(self, batch, batch_idx):
        """the full validation loop"""
        x, y = batch[0], batch[1]
        pred = self(x)
        loss = self.loss(pred, y)
        acc = self.acc(pred, y)

        return {'val_loss': loss, 'val_acc': acc}

    # OPTIONAL
    def training_epoch_end(self, outputs):
        """log and display average train loss and accuracy across epoch"""
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()

        print(f"| Train_acc: {avg_acc:.2f}, Train_loss: {avg_loss:.2f}" )

        self.log('train_loss', avg_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('train_acc', avg_acc, prog_bar=True, on_epoch=True, on_step=False)

    # OPTIONAL
    def validation_epoch_end(self, outputs):
        """log and display average val loss and accuracy"""
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        print(f"[Epoch {self.trainer.current_epoch:3}] Val_acc: {avg_acc:.2f}, Val_loss: {avg_loss:.2f}", end= " ")

        self.log('val_loss', avg_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('val_acc', avg_acc, prog_bar=True, on_epoch=True, on_step=False)

def train_classifier(train_gt, train_img_dir, fast_train=True):
    if fast_train == False:
        checkpoint = torch.load('birds_model.ckpt', map_location=torch.device('cpu'))
        model = Model()
        model.load_state_dict(checkpoint)
        return model
    else:
        model = Model()

        dataset = MyDataset(train_img_dir, train_gt, transform=transform_train)
        dataloader = DataLoader(dataset=dataset, batch_size=8, shuffle=True)


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), amsgrad=True, lr=3.0e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        for epoch in range(1):
            # Each epoch has a training and validation phase
            for phase in ['train']:
                if phase == 'train':
                    dataloader = dataloader
                    model.train()  # Set model to training mode

                running_loss = 0.
                
                # Iterate over data.
                for sample in dataloader:
                    inputs = sample[0].to(device)
                    labels = sample[1].to(device)

                    optimizer.zero_grad()

                    # forward and backward
                    with torch.set_grad_enabled(phase == 'train'):
                        preds = model(inputs)
                        loss_value = loss(preds, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss_value.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss_value.item()
                if phase == 'train':
                    scheduler.step()
        return model


swap_dict = {
    0: 0,
    1: 1,
    2: 10,
    3: 11,
    4: 12,
    5: 13,
    6: 14,
    7: 15,
    8: 16,
    9: 17,
    10: 18,
    11: 19,
    12: 2,
    13: 20,
    14: 21,
    15: 22,
    16: 23,
    17: 24,
    18: 25,
    19: 26,
    20: 27,
    21: 28,
    22: 29,
    23: 3,
    24: 30,
    25: 31,
    26: 32,
    27: 33,
    28: 34,
    29: 35,
    30: 36,
    31: 37,
    32: 38,
    33: 39,
    34: 4,
    35: 40,
    36: 41,
    37: 42,
    38: 43,
    39: 44,
    40: 45,
    41: 46,
    42: 47,
    43: 48,
    44: 49,
    45: 5,
    46: 6,
    47: 7,
    48: 8,
    49: 9
}

def classify_simple(model_filename, test_img_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model()
    model.load_state_dict(torch.load(model_filename))
    model = model.to(device)
    model.eval()

    batch_size = 16
    dataset = MyDataset(path_x=test_img_dir, transform=transform_train)
    test_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    out = {}
    for sample in tqdm(test_dataloader):
        images = sample[0]
        names = sample[2]
        preds = model(images.to(device))
        preds = preds.detach().cpu().numpy().argmax(axis=1)

        for i in range(len(sample[0])):
            out[names[i]] = swap_dict[int(preds[i])]
    
    return out

def classify(model_filename, test_img_dir, n_repeats=3):
    results = []
    for _ in range(n_repeats):
        pred = classify_simple(model_filename, test_img_dir)
        results.append(pred)
    merged = {}
    for k in results[0]:
        preds = [result[k] for result in results]
        most_common = max(set(preds), key=preds.count)
        merged[k] = most_common
    return merged
         
    