#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import time
import random

import numpy as np
from PIL import Image

import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import datasets, transforms


class DRIVE_Dataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        normal, drowsy, call, tobacco = [], [], [], []
        normal.extend(list(sorted((f'{root}/normal/01/{f}',0) for f in os.listdir(os.path.join(root, 'normal/01')) if not f.startswith('.'))))
        normal.extend(list(sorted((f'{root}/normal/08/{f}',0) for f in os.listdir(os.path.join(root, 'normal/08')) if not f.startswith('.'))))
        normal.extend(list(sorted((f'{root}/normal/15/{f}',0) for f in os.listdir(os.path.join(root, 'normal/15')) if not f.startswith('.'))))
        normal.extend(list(sorted((f'{root}/normal/20/{f}',0) for f in os.listdir(os.path.join(root, 'normal/20')) if not f.startswith('.'))))
        
        drowsy.extend(list(sorted((f'{root}/drowsy/03/{f}',1) for f in os.listdir(os.path.join(root, 'drowsy/03')) if not f.startswith('.'))))
        drowsy.extend(list(sorted((f'{root}/drowsy/10/{f}',1) for f in os.listdir(os.path.join(root, 'drowsy/10')) if not f.startswith('.'))))
        drowsy.extend(list(sorted((f'{root}/drowsy/17/{f}',1) for f in os.listdir(os.path.join(root, 'drowsy/17')) if not f.startswith('.'))))
        drowsy.extend(list(sorted((f'{root}/drowsy/22/{f}',1) for f in os.listdir(os.path.join(root, 'drowsy/22')) if not f.startswith('.'))))
        
        drowsy.extend(list(sorted((f'{root}/drowsy/02/{f}',1) for f in os.listdir(os.path.join(root, 'drowsy/02')) if not f.startswith('.'))))
        drowsy.extend(list(sorted((f'{root}/drowsy/09/{f}',1) for f in os.listdir(os.path.join(root, 'drowsy/09')) if not f.startswith('.'))))
        drowsy.extend(list(sorted((f'{root}/drowsy/16/{f}',1) for f in os.listdir(os.path.join(root, 'drowsy/16')) if not f.startswith('.'))))
        drowsy.extend(list(sorted((f'{root}/drowsy/21/{f}',1) for f in os.listdir(os.path.join(root, 'drowsy/21')) if not f.startswith('.'))))
        
        
        call.extend(list(sorted((f'{root}/call/05/{f}',2) for f in os.listdir(os.path.join(root, 'call/05')) if not f.startswith('.'))))
        call.extend(list(sorted((f'{root}/call/12/{f}',2) for f in os.listdir(os.path.join(root, 'call/12')) if not f.startswith('.'))))
        call.extend(list(sorted((f'{root}/call/19/{f}',2) for f in os.listdir(os.path.join(root, 'call/19')) if not f.startswith('.'))))
        
        tobacco.extend(list(sorted((f'{root}/tobacco/07/{f}',3) for f in os.listdir(os.path.join(root, 'tobacco/07')) if not f.startswith('.'))))
        tobacco.extend(list(sorted((f'{root}/tobacco/14/{f}',3) for f in os.listdir(os.path.join(root, 'tobacco/14')) if not f.startswith('.'))))
        
        self.imgs = normal + drowsy + call + tobacco
        random.Random(2022).shuffle(self.imgs)
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path, target = self.imgs[idx]
        img = Image.open(img_path).convert("L")
        img = np.array(img)[200:]
        img = Image.fromarray(img)
        
        if self.transforms is not None: 
            img = self.transforms(img)

        
        return img, target
    
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
          image = t(image)
        return image


def get_transform(train):
    transform_list = [transforms.ToTensor(),
                      transforms.CenterCrop((720)),
                      transforms.Resize((224)),
                      transforms.Normalize((0.5, ), (0.5, ))]
    if train:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
  
    return Compose(transform_list) 

