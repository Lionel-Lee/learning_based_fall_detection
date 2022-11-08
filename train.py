from utils.args import arg_parse
from model.fall_detection_lstm import Fall_Detection_LSTM
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
        traj_labels = torch.randint(2,(N,T,1))

        net = Fall_Detection_LSTM(dim, args.embed_size, args.lstm_hidden_size, args.lstm_num_layers)
        traj_features = net(traj_batch)
        ipdb.set_trace()