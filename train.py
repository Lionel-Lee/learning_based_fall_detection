from utils.args import arg_parse
from model.fall_detection_lstm import Fall_Detection_LSTM
from utils.traj_data_loader import MINI_Traj_Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

import ipdb

if __name__ == '__main__':
    args = arg_parse()
    if args.mode == 'train':
        N = args.batch_size
        T = args.obs_seq_len
        dim = args.motion_dim
        #random fake inputs, need dataloader for real imu data
        traj_batch = torch.randn((N,T,dim)) + torch.ones((N,T,dim))
        traj_labels = torch.randint(2,(N,1)).to(torch.float)

        net = Fall_Detection_LSTM(dim, args.embed_size, args.lstm_hidden_size, args.lstm_num_layers)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        bce_loss_func = torch.nn.BCELoss()
        for epoch in range(args.num_epoch):
            net.train()
            # # need to replace with real data
            # traj_dataset = MINI_Traj_Dataset()
            # MINI_Traj_data_loader = DataLoader(dataset=traj_dataset, batch_size = N, shuffle = True, drop_last=False)
            # for _, batch in enumerate(MINI_Traj_data_loader):
            #     pass
            traj_features = net(traj_batch)
            fall_predict = 1.*(traj_features > 0.5)
            acc = torch.mean(1.*(fall_predict == traj_labels))

            loss = bce_loss_func(traj_features, traj_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch % args.info_per_epoch == 0):
                print(f"At epoch #{epoch}, loss = {loss}, accuracy = {acc}")
