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
        self.break_val = 9999.
        self.data_src = [list(map(float, line.split())) for line in lines]

        self.data = []
        for i in range(len(self.data_src) - obs_seq_len):
            temp = self.data_src[i:i+obs_seq_len]
            if not [9999.] in temp:
                self.data.append(temp)
        self.N = len(self.data)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        imu_data_tensor = torch.tensor(self.data[index])
        # print(imu_data_tensor.shape)

        return imu_data_tensor[:,:-1], torch.tensor(int(torch.sum(imu_data_tensor[:,-1]) > self.obs_seq_len//2)).unsqueeze(dim=0).to(torch.float)

if __name__ == '__main__':
    #data loader test
    traj_dataset = MINI_Traj_Dataset(data_file_path = 'data/imu.txt', obs_seq_len = 12)
    MINI_Traj_data_loader = DataLoader(dataset=traj_dataset, batch_size = 32, shuffle = True, drop_last=False)
    print(len(MINI_Traj_data_loader))
    for _, batch in enumerate(MINI_Traj_data_loader):
        data_batch, label_batch = batch
        print(data_batch.shape)
        print(label_batch.shape)
