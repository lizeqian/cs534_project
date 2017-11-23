import numpy as np
import torch
from torchvision import models, transforms
from torch.autograd import Variable
import datetime
import cv2
import sys
import os
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
import torch.optim as optim
from logger import Logger
from network import Net

class Rand_num(Dataset):
    def __init__(self):
        dirs = []
        self.label = []
        for i in range(3):
            dirs.append(sorted(os.listdir("data/dataset/"+str(i))))
            length = len(dirs[i])
            self.label.append(np.ones(length)*i)
        for i in range(3):
            for j in range(len(dirs[i])):
                dirs[i][j] = os.path.join("data/dataset/"+str(i),dirs[i][j])
        self.directories = np.concatenate((dirs[0],dirs[1],dirs[2]))
        self.label = np.concatenate((self.label[0], self.label[1], self.label[2]))
        assert len(self.directories) == len(self.label)

    def __getitem__(self, index):
        dirs = self.directories[index]
        data = []
        for element in sorted(os.listdir(dirs)):
            dirs_mod = os.path.join(dirs,element)
            img=cv2.imread(dirs_mod,1)
            img=cv2.resize(img,None,fx=227.0/480, fy=227.0/270, interpolation = cv2.INTER_CUBIC)
            data.append(np.swapaxes(np.swapaxes(img, 2, 1), 1, 0))
        return data, self.label[index]

    def __len__(self):
        return len(self.directories)

if __name__ == '__main__':
    #####Please comment out the following 2 lines for cpu use################
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    #torch.backends.cudnn.benchmark = True

    dataset = Rand_num()
    sampler = RandomSampler(dataset)
    loader = DataLoader(dataset, batch_size = 1, sampler = sampler, shuffle = False, num_workers=2)
    a = dataset.__getitem__(0)
    net = Net()
    ########comment oput for cpu#############
    net.cuda()
    for i, data in enumerate(loader, 0):
        video, labels = data
        labels = Variable(labels.long().cuda())
        for j, frame in enumerate(video):
    #############CPU###############
#            frame = Variable(frame.float()/256)
            frame = Variable((frame.float()/256).cuda())
            outputs = net.forward(frame)
            print outputs
            print labels
            loss = net.lossFunction(outputs, labels)
            print loss
            break
        break

