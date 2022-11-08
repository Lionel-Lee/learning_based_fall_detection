from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
# import numpy as np
# import pandas as pd
# import math

# import torch
# import torch.nn.functional as F
# import torchvision
# import torchvision.transforms as T

class MINI_Traj_Dataset(Dataset):

    def __init__(self, data_file_path, obs_seq_len):
        super(MINI_Traj_Dataset, self).__init__()
        data_file = open(data_file_path, 'r')
        lines = data_file.readlines()
        self.obs_seq_len = obs_seq_len
        self.data = [list(map(float, line.split())) for line in lines]
        self.N = len(self.data) - obs_seq_len + 1

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return torch.tensor(self.data[index:index+self.obs_seq_len])

if __name__ == '__main__':
    #data loader test
    traj_dataset = MINI_Traj_Dataset(data_file_path = 'data/imu.txt', obs_seq_len = 12)
    MINI_Traj_data_loader = DataLoader(dataset=traj_dataset, batch_size = 32, shuffle = True, drop_last=False)
    print(len(MINI_Traj_data_loader))
    for _, batch in enumerate(MINI_Traj_data_loader):
        print(batch.shape)
