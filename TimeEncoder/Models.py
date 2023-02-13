import torch
import torch.nn as nn


## input_dim : features

class Recovery(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_out, n_layers, batch_size, device):
        super(Recovery, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_out = n_out
        self.n_layers = n_layers
        self.device = device
        self.batch_size = batch_size

        self.gru = nn.GRU(self.input_dim, self.hidden_dim, self.n_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.n_out)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        #out = torch.flatten(out)
        #print(out.size())
        out = self.fc(self.sigmoid(out[:,-1]))
        return(out, h)

    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_().to(self.device)
        return(hidden)

class Embedder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, batch_size, device):
        super(Embedder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_out = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.batch_size = batch_size

        self.gru = nn.GRU(self.input_dim, self.hidden_dim, self.n_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.n_out)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.sigmoid(out))#[:,-1]))
        return(out, h)

    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_().to(self.device)
        return(hidden)

class TimeEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_out, n_layers, batch_size, device):
        super(TimeEncoder, self).__init__()
        self.Embedder = Embedder(input_dim, hidden_dim, n_layers, batch_size, device)
        self.Recovery = Recovery(input_dim, hidden_dim, n_out, n_layers, batch_size, device)

    def forward(self, x, h):
        #print(x.size(), h.size())
        out, h = self.Embedder(x, h)
        #print(out.size(), self.h.data.size())
        out, self.h = self.Recovery(out, self.h.data)
        #print()
        return(out, self.h)

    def init_hidden(self):
        self.h = self.Recovery.init_hidden()
        return(self.Embedder.init_hidden())
