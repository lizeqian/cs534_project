import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.autograd as autograd

class LRCN(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, lstm_layer, batch):
        super(LRCN, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch = batch
        self.num_layers = lstm_layer
        self.lstm = nn.LSTM(256*6*6, hidden_dim, lstm_layer, dropout = 0.5)
        self.hidden = self.init_hidden()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, embedding_dim),
        )
        self.final_linear = nn.Linear(hidden_dim, 3)
        self.sm = nn.Softmax()

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(self.num_layers, self.batch, self.hidden_dim)),
                autograd.Variable(torch.zeros(self.num_layers, self.batch, self.hidden_dim)))

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.batch, 256 * 6 * 6)
        #x = self.classifier(x)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        outs = lstm_out[-1]
        outs = self.final_linear(outs)
        #outs = self.sm(outs)
        return outs





