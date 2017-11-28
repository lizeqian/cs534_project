import numpy as np
import torch
from torchvision import models, transforms
from torch.autograd import Variable
import datetime
import sys
import os
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
import torch.optim as optim
from logger import Logger
from lstmlayer import LSTMLayer
import torch.nn as nn

class Rand_num(Dataset):
    def __init__(self):
        self.dirs=sorted(os.listdir("data/cnndata/"))
        for j in range(len(self.dirs)):
            self.dirs[j] = os.path.join("data/cnndata/",self.dirs[j])

    def __getitem__(self, index):
        file_name = self.dirs[index]
        data = np.loadtxt(file_name,delimiter=',', skiprows=1)
        label = np.genfromtxt(file_name,delimiter=',', usecols = 0)

        return data, label[0]

    def __len__(self):
        return len(self.dirs)

if __name__ == '__main__':
    #####Please comment out the following 2 lines for cpu use################
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.backends.cudnn.benchmark = True

    lossfunction = nn.CrossEntropyLoss()

    dataset = Rand_num()
    sampler = RandomSampler(dataset)
    loader = DataLoader(dataset, batch_size = 1, sampler = sampler, shuffle = False, num_workers=1)
    net = LSTMLayer(1000, 3, 5)
    net.cuda()
    optimizer = optim.Adam(net.parameters(), lr=0.00001)
    for epoch in range(1000):
        for i, data in enumerate(loader, 0):
            net.zero_grad()
            net.hidden = net.init_hidden()
            video, labels = data
            labels = torch.squeeze(Variable(labels.long().cuda()))
            video = torch.squeeze(Variable((video.float()/256).cuda()))
            net.train()
            outputs = net.forward(video)
            loss = lossfunction(outputs, labels)
            loss.backward()
            optimizer.step()
            if i == 0:
                print (loss)
