import numpy as np
import torch
from torchvision import models, transforms
from torch.autograd import Variable
import datetime
import sys
import os
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
import torch.optim as optim
from logger import Logger
from lstmlayer import LSTMLayer
import torch.nn as nn

class Rand_num(Dataset):
    def __init__(self):
        self.dirs=sorted(os.listdir("data/cnndata_test/"))
        for j in range(len(self.dirs)):
            self.dirs[j] = os.path.join("data/cnndata_test/",self.dirs[j])

    def __getitem__(self, index):
        file_name = self.dirs[index]
        data = np.loadtxt(file_name,delimiter=',', skiprows=1)
        label = np.genfromtxt(file_name,delimiter=',', usecols = 0)
        return data[0:50], label[0]

    def __len__(self):
        return len(self.dirs)

if __name__ == '__main__':
    #####Please comment out the following 2 lines for cpu use################
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.backends.cudnn.benchmark = True

    SAVE_PATH = './cp_lstm_128.bin'

    lossfunction = nn.CrossEntropyLoss()
    batch_size = 1

    dataset = Rand_num()
    sampler = SequentialSampler(dataset)
    loader = DataLoader(dataset, batch_size, sampler = sampler, shuffle = False, num_workers=1, drop_last=True)
    net = LSTMLayer(1000, 32, 1, batch_size)
    net.load_state_dict(torch.load(SAVE_PATH))
    net.cuda()
    for i, data in enumerate(loader, 0):
        video, labels = data
        video = Variable((video.float()/256).cuda()).permute(1,0,2)
        labels = Variable(labels.float().cuda())
        net.hidden = net.init_hidden()
        net.eval()
        outputs = net.forward(video)
        _, maxout = torch.max(outputs, 1)
        accu = torch.mean(torch.eq(labels, maxout.float()).float())
        print(labels.data.cpu().numpy().item(), maxout.data.cpu().numpy().item())
