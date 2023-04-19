import torch
import torch.nn as nn


class SMAPELoss(nn.Module):
    def __init__(self):
        super(SMAPELoss, self).__init__()

    def forward(self, y, y_hat):
        delta_y = torch.abs(torch.sub(y, y_hat, alpha=1))
        scale = torch.add(torch.abs(y), torch.abs(y_hat), alpha=1)
        smape = torch.div(delta_y, scale)
        smape = torch.nan_to_num(smape, nan=0.0, posinf=0.0, neginf=0.0)
        smape = torch.mul(torch.mean(smape), 2)
        return(smape)


class WMAPELoss(nn.Module):
    def __init__(self):
        super(WMAPELoss, self).__init__()

    def forward(self, y, y_hat):
        delta_y = torch.sum(torch.abs(torch.sub(y, y_hat, alpha=1)))
        scale = torch.sum(torch.abs(y))
        wmape = torch.div(delta_y, scale)
        wmape = torch.nan_to_num(wmape, nan=10.0, posinf=10.0, neginf=-10.0)
        return(wmape)
