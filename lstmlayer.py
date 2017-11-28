import torch
import torch.nn as nn
import torch.autograd as autograd

class LSTMLayer(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, lstm_layer):
        self.hidden_dim = hidden_dim
        self.num_layers = lstm_layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, lstm_layer)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(self.num_layers, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(self.num_layers, 1, self.hidden_dim)))

    def forward(self, inputs):
        inputs = inputs.view(inputs.size()[0], 1, inputs.size()[1])
        lstm_out, self.hidden = self.lstm(inputs, self.hidden)

