import torch
import torch.nn as nn

class Estimator(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_out, n_layers, batch_size, device, model_type, seq_len):
        super(Estimator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_out = n_out
        self.n_layers = n_layers
        self.device = device
        self.batch_size = batch_size
        self.model_type = model_type
        self.seq_len = seq_len
        self.dropout_prob = 0.0

        if self.model_type=='GRU':
            self.main = nn.GRU(self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True, dropout=self.dropout_prob)
        else:
            self.main = nn.LSTM(self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True, dropout=self.dropout_prob)
        self.fc = nn.Linear(self.hidden_dim*self.seq_len, self.n_out,device=self.device)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.tanh = nn.Tanh()


    def forward(self, x, h):
        out, h = self.main(x, h)
        out = self.fc(self.tanh(torch.flatten(out, start_dim=1)))
        out = self.dropout(out)
        out = torch.reshape(out, (x.size()[0],1))
        return(out, h)

    def init_hidden(self):
        weight = next(self.parameters()).data
        if self.model_type=='GRU':
            hidden = weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_().requires_grad_().to(self.device)
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
        self.dropout_prob = 0.0

        if self.model_type=='GRU':
            self.main = nn.GRU(self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True, dropout=self.dropout_prob)
        else:
            self.main = nn.LSTM(self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True, dropout=self.dropout_prob)
        self.fc = nn.Linear(self.hidden_dim, self.n_out,device=self.device)
        self.fc_norm = nn.Linear(self.input_dim, self.hidden_dim, device=self.device)

        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(self.dropout_prob)

    def forward(self, x, h):
        x = self.fc_norm(x)
        x = self.sigmoid(x)
        x = self.dropout(x)
        out, h = self.main(x, h)
        out = self.sigmoid(self.fc(out))
        out = self.dropout(out)
        out = torch.add(out,x,alpha=1)
        return(out, h)

    def init_hidden(self):
        weight = next(self.parameters()).data

        if self.model_type=='GRU':
            hidden = weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_().requires_grad_().to(self.device)
        else:
            hidden = (weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_().to(self.device),
                  weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_().to(self.device))
        return(hidden)
