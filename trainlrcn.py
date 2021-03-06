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
from lrcn import LRCN
import torch.nn as nn

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
        img_dir = sorted(os.listdir(dirs))
        data = []
        for i in range(50):
            dirs_mod = os.path.join(dirs,img_dir[i])
            img=cv2.imread(dirs_mod,1)
            img=cv2.resize(img,None,fx=227.0/480, fy=227.0/270, interpolation = cv2.INTER_CUBIC)
            data.append(np.swapaxes(np.swapaxes(img, 2, 1), 1, 0))

            outlabel = np.zeros(3)
            outlabel[int(self.label[index])] = 1
        return np.array(data), self.label[index]

    def __len__(self):
        return len(self.directories)

if __name__ == '__main__':
    #####Please comment out the following 2 lines for cpu use################
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.backends.cudnn.benchmark = True
    batch_size = 10
    SAVE_PATH = './cp_lrcn2.bin'
    lossfunction = nn.CrossEntropyLoss()
    logger = Logger('./lrcnlogs2')

    dataset = Rand_num()
    sampler = RandomSampler(dataset)
    loader = DataLoader(dataset, batch_size, sampler = sampler, shuffle = False, num_workers=1, drop_last=True)
    net = LRCN(1000, 128, 5, batch_size)
    #net.load_state_dict(torch.load(SAVE_PATH))
    net.cuda()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    for epoch in range(10000):
        for i, data in enumerate(loader, 0):
            video, labels = data
            video = video.permute(1,0,2,3,4)
            video = video.contiguous().view(-1,3,227,227)
            net.zero_grad()
            net.hidden = net.init_hidden()
            labels = Variable(labels.long().cuda())
            video = Variable((video.float()/256).cuda())
            net.train()
            outputs = net.forward(video)
            loss = lossfunction(outputs, labels)
            loss.backward()
            optimizer.step()
            if i == 0:
                print(datetime.datetime.now())
                torch.save(net.state_dict(), SAVE_PATH)
                net.eval()
                outputs = net.forward(video)
                _, maxout = torch.max(outputs, 1)
                #_, gtlabel = torch.max(labels, 1)
                accu = torch.mean(torch.eq(labels.float(), maxout.float()).float())
                print (loss)
                print (accu)
                logger.scalar_summary('loss', loss.data.cpu().numpy(), epoch)
                logger.scalar_summary('accu', accu.data.cpu().numpy(), epoch)

