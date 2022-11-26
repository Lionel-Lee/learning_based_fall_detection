from utils.args import arg_parse
from model.fall_detection_lstm import Fall_Detection_LSTM
from utils.traj_data_loader import MINI_Traj_Dataset
from torch.utils.data import DataLoader
# from torch.autograd import Variable
import torch
import os
import time
from torch.optim.lr_scheduler import StepLR

import ipdb

if __name__ == '__main__':
    args = arg_parse()
    N = args.batch_size
    T = args.obs_seq_len
    dim = args.motion_dim
    bce_loss_func = torch.nn.BCELoss()
    # for testing
    torch.manual_seed(0)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    
    device = torch.device("cpu") 

    # python main.py --mode train --data_file_path data/imu_train.txt
    if args.mode == 'train':
        net = Fall_Detection_LSTM(dim, args.embed_size, args.lstm_hidden_size, args.lstm_num_layers)
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        if args.lr_scheduler == 'StepLR':
            lr_scheduler = StepLR(optimizer, step_size=args.num_epoch//4, gamma=args.lr_scheduler_gamma)

        for epoch in range(args.num_epoch):
            net.train()
            traj_dataset = MINI_Traj_Dataset(data_file_path = args.data_file_path, obs_seq_len = args.obs_seq_len)
            MINI_Traj_data_loader = DataLoader(dataset=traj_dataset, batch_size = N, shuffle = True, drop_last=True)

            avg_loss = 0.
            avg_acc = 0.
            num_batch = 1.*len(MINI_Traj_data_loader)
            for _, batch in enumerate(MINI_Traj_data_loader):
                #random fake inputs for labels, need dataloader for real imu data
                traj_batch, traj_labels = batch
                traj_batch = traj_batch.to(device)
                traj_labels = traj_labels.to(device)

                traj_features = net(traj_batch)
                fall_predict = 1.*(traj_features > 0.5)
                acc = torch.mean(1.*(fall_predict == traj_labels))

                loss = bce_loss_func(traj_features, traj_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                avg_loss += loss
                avg_acc += acc

            if (epoch % args.info_per_epoch == 0):
                print(f"At epoch #{epoch}, avg_loss = {avg_loss/num_batch}, avg_accuracy = {avg_acc/num_batch}")
            if args.lr_scheduler is not None:
                lr_scheduler.step()
        torch.save(net.state_dict(), args.model_save_path)

    # python main.py --mode eval --data_file_path data/imu_eval.txt
    elif args.mode == 'eval':        
        net = Fall_Detection_LSTM(dim, args.embed_size, args.lstm_hidden_size, args.lstm_num_layers)
        net.load_state_dict(torch.load(args.model_save_path))
        net.eval()

        traj_dataset = MINI_Traj_Dataset(data_file_path = args.data_file_path, obs_seq_len = args.obs_seq_len)
        MINI_Traj_data_loader = DataLoader(dataset=traj_dataset, batch_size = N, shuffle = True, drop_last=True)

        with torch.no_grad():
            avg_loss = 0.
            avg_acc = 0.
            num_batch = 1.*len(MINI_Traj_data_loader)
            for _, batch in enumerate(MINI_Traj_data_loader):
                #random fake inputs for labels, need dataloader for real imu data
                traj_batch, traj_labels = batch    
                traj_batch = traj_batch.to(device)
                traj_labels = traj_labels.to(device)

                traj_features = net(traj_batch)
                fall_predict = 1.*(traj_features > 0.5)
                acc = torch.mean(1.*(fall_predict == traj_labels))

                loss = bce_loss_func(traj_features, traj_labels)
                avg_loss += loss
                avg_acc += acc
            
            print(f"avg_loss = {avg_loss/num_batch}, avg_accuracy = {avg_acc/num_batch}")

    elif args.mode == 'deploy':

        while (True):
            time.sleep(0.2)
            ret = os.system('sshpass -p \"1234\" scp mini@192.168.102.32:/home/mini/test.txt /home/lionel/ECE598JK/learning_based_fall_detection/data')
            assert ret == 0