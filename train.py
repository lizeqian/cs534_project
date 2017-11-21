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
import torch.optim as optim
from logger import Logger

class Rand_num(Dataset):
    def __init__(self):   
        self.dirs = []
        self.label = []
        for i in range(3):
            self.dirs.append(sorted(os.listdir("data/dataset/"+str(i))))
            length = len(self.dirs[i])
            self.label.append(np.ones(length)*i)
        for i in range(3):
            for j in range(len(self.dirs[i])):
                self.dirs[i][j] = os.path.join("data/dataset/"+str(i),self.dirs[i][j])
        
        self.data = []
        
        for i in range(1):
            for j in range(1):
                sample = []
                for dir in sorted(os.listdir(self.dirs[i][j])):
                    dirs_mod = os.path.join(self.dirs[i][j],dir)
                    print(dirs_mod)
                    sample.append(cv2.imread(dirs_mod,1))
                self.data.append(sample)
                
        self.label = np.concatenate(self.label)    
            
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
    
if __name__ == '__main__':
    dataset = Rand_num();
    a=dataset.data