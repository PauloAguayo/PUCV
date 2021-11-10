import torch
import numpy as np
from torch.utils.data import Dataset

def format_training_data(pet_names, pet_labels, window, device):
    characters = [1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,20.]

    pet_names_numeric = []
    for i in range(len(pet_names)):
        pet_names_numeric.append([[pet_names[i,j*51:(j+1)*51]] for j in range(window)])

    pet_labels_numeric = [characters.index(k) for k in pet_labels]

    y = torch.tensor(pet_labels_numeric, device=device)
    x = torch.tensor(pet_names_numeric, device=device)

    return(x, y)

class OurDataset(Dataset):
    def __init__(self, pet_names,pet_labels,window, device):
        self.x, self.y = format_training_data(pet_names,pet_labels,window,device)
        self.permute()

    def __getitem__(self, idx):
        idx = self.permutation[idx]
        return(self.x[idx], self.y[idx])

    def __len__(self):
        return(len(self.x))

    def permute(self):
        self.permutation = torch.randperm(len(self.x))
