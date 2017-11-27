from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import pdb
from IPython.core.debugger import Tracer
import math

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
                            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=3, stride=2),
                            nn.Conv2d(64, 192, kernel_size=5, padding=2),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=3, stride=2),
                            nn.Conv2d(192, 384, kernel_size=3, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(384, 256, kernel_size=3, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(256, 256, kernel_size=3, padding=1),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=3, stride=2),
                        )

        self.hidden_dimension = 3
        self.hidden_layer = 1
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.hidden_dimension),
        )
        self.lstm = nn.LSTM(self.hidden_dimension, self.hidden_dimension, self.hidden_layer)
        self.hidden = self.init_hidden()

        self.linear_final = nn.Linear(self.hidden_dimension, 3)
        self.softmax_final = nn.Softmax()

    def init_hidden(self):
        return (Variable((torch.randn(self.hidden_layer, 1, self.hidden_dimension)).float().cuda()), Variable((torch.randn(self.hidden_layer, 1, self.hidden_dimension)).float().cuda()))


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], 256 * 6 * 6)
        x = self.classifier(x)
        #lstm_out, self.hidden = self.lstm(x.view(x.size()[0], 1, -1), self.hidden)
        #lstm_out = torch.squeeze(lstm_out)
        #x = self.linear_final(lstm_out)
        x = self.softmax_final(x)
        return x

    def lossFunction(self, predicts, labels):
        loss = nn.CrossEntropyLoss()
        output = loss(predicts, labels)
        return output

