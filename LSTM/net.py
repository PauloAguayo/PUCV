import torch
import torch.nn as nn



class Net(nn.Module):
    def __init__(self, input_size=51, hidden_size=100, n_layers=1, n_class=12, device=None):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_size, n_class),
                                nn.Softmax(dim=1))


    def forward(self,x):
        x = torch.squeeze(x).float()
        out, _ = self.lstm(x)
        out = out[:,-1,:]
        out = self.fc(out)
        return(out)
