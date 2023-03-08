import torch
import torch.nn as nn


## input_dim : features

class Recovery(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_out, n_layers, batch_size, device, model_type):
        super(Recovery, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_out = n_out
        self.n_layers = n_layers
        self.device = device
        self.batch_size = batch_size
        self.model_type = model_type

        if self.model_type=='GRU':
            self.main = nn.GRU(self.input_dim, self.hidden_dim, self.n_layers, batch_first=True)
        else:
            self.main = nn.LSTM(self.input_dim, self.hidden_dim, self.n_layers, batch_first=True)

        self.fc = nn.Linear(self.hidden_dim, self.n_out)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h):
        out, h = self.main(x, h)
        out = self.fc(self.sigmoid(out[:,-1])) ############################################33
        return(out, h)

    def init_hidden(self):
        weight = next(self.parameters()).data
        if self.model_type=='GRU':
            hidden = weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_().to(self.device)
        else:
            hidden = (weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_().to(self.device),
                  weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_().to(self.device))
        return(hidden)

class Embedder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, batch_size, device, model_type):
        super(Embedder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_out = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.batch_size = batch_size
        self.model_type = model_type

        if self.model_type=='GRU':
            self.main = nn.GRU(self.input_dim, self.hidden_dim, self.n_layers, batch_first=True)
        else:
            self.main = nn.LSTM(self.input_dim, self.hidden_dim, self.n_layers, batch_first=True)

        self.fc = nn.Linear(self.hidden_dim, self.n_out)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h):
        out, h = self.main(x, h)
        out = self.fc(self.sigmoid(out))
        return(out, h)

    def init_hidden(self):
        weight = next(self.parameters()).data
        if self.model_type=='GRU':
            hidden = weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_().to(self.device)
        else:
            hidden = (weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_().to(self.device),
                  weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_().to(self.device))
        return(hidden)

class TimeEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_out, n_layers, batch_size, device, model_type):
        super(TimeEncoder, self).__init__()
        self.model_type = model_type
        self.Embedder = Embedder(input_dim, hidden_dim, n_layers, batch_size, device, self.model_type)
        self.Recovery = Recovery(input_dim, hidden_dim, n_out, n_layers, batch_size, device, self.model_type)

    def forward(self, x, h):
        out, h = self.Embedder(x, h)

        if self.model_type=="GRU":
            out, self.h = self.Recovery(out, self.h.data)
        else:
            self.h = tuple([e.data for e in self.h])
            out, self.h = self.Recovery(out, self.h)
        return(out, self.h)

    def init_hidden(self):
        self.h = self.Recovery.init_hidden()
        return(self.Embedder.init_hidden())
