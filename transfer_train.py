import numpy as np
import torch
from torchvision import models, transforms
import torch.utils.model_zoo as model_zoo
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
from model_zoo import AlexNet
from model_zoo import VGG
import torch.nn as nn
from lstmlayer import LSTMLayer

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
	'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class Rand_num(Dataset):
    def __init__(self, datadir):
        dirs = []
        self.label = []
        for i in range(3):
            dirs.append(sorted(os.listdir(datadir+str(i))))
            length = len(dirs[i])
            self.label.append(np.ones(length)*i)
        for i in range(3):
            for j in range(len(dirs[i])):
                dirs[i][j] = os.path.join(datadir+str(i),dirs[i][j])
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
    #####params set up################
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.backends.cudnn.benchmark = True
    batch_size = 10
    lossfunction = nn.CrossEntropyLoss()

    model_selection = 'alex'

    if model_selection == 'vgg':
        logger = Logger('./vgglogs')
        SAVE_PATH0 = './cp_vgg.bin'
        SAVE_PATH1 = './cp_vgg.bin'
    else:
        logger = Logger('./alexlogs')
        SAVE_PATH0 = './cp_alex.bin'
        SAVE_PATH1 = './cp_alex.bin'


	#########Dataset set up###########
    dataset = Rand_num("data/dataset/")
    sampler = RandomSampler(dataset)
    loader = DataLoader(dataset, batch_size, sampler = sampler, shuffle = False, num_workers=1, drop_last=True)

	##########Network set up##########
    lstmnet = LSTMLayer(4096, 64, 2, batch_size)

    if model_selection == 'vgg':
    	model = VGG()
    else:
    	model = AlexNet()

	################Load dict#############
    if model_selection == 'vgg':
    	pretrained_dict = model_zoo.load_url(model_urls['vgg11_bn'])
    else:
    	pretrained_dict = model_zoo.load_url(model_urls['alexnet'])

    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    lstmnet.cuda()
    model.cuda()
    optimizer = optim.Adam([{'params': model.parameters()}, {'params': lstmnet.parameters(), 'lr': 0.0001}], lr=0.000001, weight_decay=0.001)
    for epoch in range(10000):
        for i, data in enumerate(loader, 0):
            video, labels = data
            video = video.permute(1,0,2,3,4)
            video = video.contiguous().view(-1,3,227,227)
            lstmnet.zero_grad()
            model.zero_grad()
            lstmnet.hidden = lstmnet.init_hidden()
            labels = Variable(labels.long().cuda())
            video = Variable((video.float()/256).cuda())
            lstmnet.train()
            model.train()
            outputs = model.forward(video)
            outputs = outputs.view(-1, batch_size, 4096)
            outputs = lstmnet.forward(outputs)
            loss = lossfunction(outputs, labels)
            loss.backward()
            optimizer.step()
            if i == 0:
                print(datetime.datetime.now())
                torch.save(lstmnet.state_dict(), SAVE_PATH0)
                torch.save(model.state_dict(), SAVE_PATH1)
                lstmnet.eval()
                model.eval()
                outputs = model.forward(video)
                outputs = lstmnet.forward(outputs)
                _, maxout = torch.max(outputs, 1)
                #_, gtlabel = torch.max(labels, 1)
                accu = torch.mean(torch.eq(labels.float(), maxout.float()).float())
                print (loss)
                print (accu)
                logger.scalar_summary('loss', loss.data.cpu().numpy(), epoch)
                logger.scalar_summary('accu', accu.data.cpu().numpy(), epoch)

