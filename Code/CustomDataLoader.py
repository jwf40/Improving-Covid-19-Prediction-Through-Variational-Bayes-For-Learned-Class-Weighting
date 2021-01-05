import os
import random
import numpy as np
from skimage import io, transform
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
import matplotlib.pyplot as plt

class CovidDataSet(torch.utils.data.Dataset):
    def __init__(self, weights=None):
        self.root_path ='./'
        self.training_dataroot = "data/train"
        self.validation_dataroot = "data/test"
        # Threads for dataloader
        self.workers = 2
        # Spatial size of training images.
        self.image_size = 36
        # image crop, matches ResNet
        self.image_crop = 32
        # Affine Transform Values
        self.degrees = (-10, 10)
        self.translate = ((1/10), (1/10))
        self.scale = (0.85, 1.15)
        self.resample = False
        self.fillcolor = 0
        self.brightness = (0.9, 1.1)
        self.weights = np.power(weights,1) if weights is not None else weights
        self.train_dataset = self.get_training_dataset()
        self.dataset = self.make_dataset()

    def __len__(self):
        return len(self.train_dataset.imgs)

    def apply_weights(self):
        for idx,_ in enumerate(self.dataset):
            self.dataset[idx].weight = self.weights[idx]

    def make_dataset(self):
        dataset = []
        with open(self.root_path+"train_split_v3.txt", 'r') as f:            
            data = f.readlines()
            if self.weights is None:
                self.weights = [1 for _ in range(self.__len__())] 
            for idx,sample in enumerate(data):
                sample = sample.split(" ")
                dataset.append({"name":sample[1], "target":sample[2], "weight":self.weights[idx]})
        return dataset

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_location = self.root_path+self.training_dataroot+ "/" + self.dataset[idx]['target'] + '/' + self.dataset[idx]['name']
        image = Image.fromarray(np.uint8(io.imread(sample_location))).convert('RGB')
        image = self.transform(image).reshape(3,32,32)
        target = self.dataset[idx]['target']
        labels = {'COVID-19': 0, 'normal': 1, 'pneumonia':2}
        sample = {"img":image, "target": torch.tensor(labels[target]), "weight":torch.tensor(self.dataset[idx]['weight'])}
        return sample

        
    def get_training_dataset(self):
        # Load training images, apply affine transforms, resize images, normalise.
        train_dataset = dset.ImageFolder(root=self.root_path+self.training_dataroot,
                                        transform=transforms.Compose([
                                                transforms.RandomHorizontalFlip(p=0.5),
                                                transforms.RandomAffine(self.degrees, translate=self.translate, scale=self.scale, resample=self.resample, fillcolor=self.fillcolor),
                                                transforms.ColorJitter(brightness=self.brightness),
                                                transforms.Resize(self.image_size),
                                                transforms.RandomCrop(self.image_crop, padding=4),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                ])
                                        )
        return train_dataset

    def transform(self,img):
        trans=transforms.Compose([
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomAffine(self.degrees, translate=self.translate, scale=self.scale, resample=self.resample, fillcolor=self.fillcolor),
                                    transforms.ColorJitter(brightness=self.brightness),
                                    transforms.Resize(self.image_size),
                                    transforms.RandomCrop(self.image_crop, padding=4),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                    ])
        return trans(img)
