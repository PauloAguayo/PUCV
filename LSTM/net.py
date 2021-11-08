import torch
import torch.nn as nn



class Net(nn.Module):
    def __init__(self, input_size=51, hidden_size=100, n_layers=1, n_class=12, device=None):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_size, n_class),
                                nn.Softmax(dim=1))
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.device = device



    def forward(self,x):
        # for e,vector_ex in enumerate(x):
        #     # self.hidden = (torch.randn(self.n_layers, 1, self.hidden_size).to(self.device),
        #     #                 torch.randn(self.n_layers, 1, self.hidden_size).to(self.device))
        #     for vector_ in vector_ex:
        #         # print(vector_)
        #         out, self.hidden = self.lstm((vector_.view(1, 1, -1)).float(),self.hidden)
        #     if e==0: out_vector = out
        #     else: out_vector = torch.vstack((out_vector, out))
        #
        # out = self.fc(out_vector)
        # return(out)
        # print(x.size())
        x = torch.squeeze(x).float()
        out, _ = self.lstm(x)
        out = out[:,-1,:]
        out = self.fc(out)
        return(out)
