import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms
from torch.autograd import Variable
import datetime
import cv2
import glob
import os
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from model.network import Net
import torch.optim as optim
from logger import Logger

class Rand_num(Dataset):
    def __init__(self):   
        data = []
        label = []
        for i in range(3):
            data.append(sorted(os.listdir("data/dataset/"+str(i))))
            length = len(data[i])
            label.append(np.ones(length)*i)
        label = np.concatenate(label)    
            
        self.classes=('0', '1', '2')

    def __getitem__(self, index):
        #print ('\tcalling Dataset:__getitem__ @ idx=%d'%index)
        img = self.images[index]
        label = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
#        print ('\tcalling Dataset:__len__')
        return len(self.images)