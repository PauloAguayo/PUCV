import torch
import numpy as np
import random
from torch.utils.data import Dataset

def format_training_data(pet_names, pet_labels, window, device, characters):

    pet_names_numeric = []
    for i in range(len(pet_names)):
        pet_names_numeric.append([[pet_names[i,j*51:(j+1)*51]] for j in range(window)])

    pet_labels_numeric = [characters.index(k) for k in pet_labels]

    y = torch.tensor(pet_labels_numeric, device=device)
    x = torch.tensor(pet_names_numeric, device=device)

    return(x, y)

class OurDataset(Dataset):
    def __init__(self, pet_names,pet_labels,window, device, characters):
        self.x, self.y = format_training_data(pet_names,pet_labels,window,device,characters)
        self.permute()

    def __getitem__(self, idx):
        idx = self.permutation[idx]
        return(self.x[idx], self.y[idx])

    def __len__(self):
        return(len(self.x))

    def permute(self):
        self.permutation = torch.randperm(len(self.x))


class randomizeDataset(object):
    def __init__(self,raw_data,true_label):
        self.raw_data = raw_data
        self.true_label = true_label

        self.rows = np.arange(len(raw_data))
        np.random.shuffle(self.rows)

        self.new_raw_data = np.zeros((len(self.rows)))
        self.new_true_label = np.zeros((len(self.rows)))

    def random_random(self):

        self.new_raw_data = np.array([self.raw_data[i] for i in self.rows])
        self.new_true_label = np.array([self.true_label[i] for i in self.rows])

        print(self.new_raw_data)
        print(self.new_true_label)

        return([self.new_raw_data,self.new_true_label])
