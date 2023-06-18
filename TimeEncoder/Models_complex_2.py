import torch
import torch.nn as nn

class Estimator(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_out, n_layers, batch_size, device, model_type, seq_len):
        super(Estimator, self).__init__()


    def init_hidden(self):
        weight = next(self.parameters()).data
        if self.model_type=='GRU':
            hidden = weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_().requires_grad_().to(self.device)
        else:
            hidden = (weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_().to(self.device),
                  weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_().to(self.device))
        return(hidden)

class Embedder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, batch_size, device, model_type,seq_len):
        super(Embedder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_out = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.batch_size = batch_size
        self.model_type = model_type
        self.seq_len = seq_len

        if self.model_type=='GRU':
            self.main = nn.GRU(self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True)
        else:
            self.main = nn.LSTM(self.input_dim, self.hidden_dim, self.n_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.n_out,device=self.device)
        self.fc_norm = nn.Linear(self.input_dim, self.hidden_dim,device=self.device)
        self.fc_final = nn.Linear(self.hidden_dim*self.seq_len,1,device=self.device)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h):
        print(x.is_leaf)
        out0 = self.fc_norm(x)
        out0.retain_grad()
        print(out0.is_leaf)
        # x = self.tanh(x)
        out1 = self.sigmoid(out0)
        out1.retain_grad()
        print(out1.is_leaf)
        out_copy = out1.clone()
        out_copy.retain_grad()
        print(out_copy.is_leaf)
        out2, h = self.main(out1, h)
        out2.retain_grad()
        print(out2.is_leaf,h.is_leaf)
        # h = torch.permute(h, (1, 0, 2))
        # out = self.tanh(self.fc(h[:,-1,:]))
        # out = self.tanh(out)
        out3 = self.sigmoid(self.fc(out2))
        out3.retain_grad()
        print(out3.is_leaf)
        # out = torch.add(out, out_copy, alpha=1)
        out4 = out3 + out_copy
        out4.retain_grad()
        print(out4.is_leaf)

        out5, h = self.main(x4, h)
        out5.retain_grad()
        out6 = self.fc_final(self.tanh(torch.flatten(out5, start_dim=1)))
        out6.retain_grad()
        out7 = torch.reshape(out6, (x.size()[0],1))
        out7.retain_grad()
        return(out7, h)

    def init_hidden(self):
        weight = next(self.parameters()).data

        if self.model_type=='GRU':
            hidden = weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_().requires_grad_().to(self.device)
        else:
            hidden = (weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_().to(self.device),
                  weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_().to(self.device))
        return(hidden)
