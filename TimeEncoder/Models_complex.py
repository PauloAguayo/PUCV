import torch
import torch.nn as nn

class Estimator(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_out, n_layers, batch_size, device, model_type):
        super(Estimator, self).__init__()
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

        self.tanh = nn.Tanh()


    def forward(self, x, h):
        # x = torch.reshape(x, (x.size()[0],1,x.size()[1]))
        out, h = self.main(x, h)
        h = torch.permute(h, (1, 0, 2))
        out = self.tanh(self.fc(h[:,-1,:]))
        # out = self.tanh(self.fc(out))
        out = torch.reshape(out, (x.size()[0],1))
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
        self.fc_norm = nn.Linear(self.input_dim, self.input_dim)

        self.tanh = nn.Tanh()

    def forward(self, x, h):
        x = self.fc_norm(x)
        x = self.tanh(x)
        out, h = self.main(x, h)
        h = torch.permute(h, (1, 0, 2))
        # out = self.tanh(self.fc(h[:,-1,:]))
        out = self.tanh(out)
        out = torch.add(out, x, alpha=1)
        return(out, h)

    def init_hidden(self):
        weight = next(self.parameters()).data

        if self.model_type=='GRU':
            hidden = weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_().to(self.device)
        else:
            hidden = (weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_().to(self.device),
                  weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_().to(self.device))
        return(hidden)
