from utils.args import arg_parse
from model.fall_detection_lstm import Fall_Detection_LSTM
from utils.traj_data_loader import MINI_Traj_Dataset
from torch.utils.data import DataLoader
# from torch.autograd import Variable
import torch
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

    if args.mode == 'train':
        net = Fall_Detection_LSTM(dim, args.embed_size, args.lstm_hidden_size, args.lstm_num_layers)
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
                traj_batch = batch
                # traj_labels = torch.randint(2,(traj_batch.shape[0],1)).to(torch.float)
                traj_labels = torch.ones((traj_batch.shape[0],1)).to(torch.float)

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

    elif args.mode == 'eval':
        #random fake inputs, need dataloader for real imu data
        traj_batch = torch.randn((N,T,dim)) + torch.ones((N,T,dim))
        traj_labels = torch.randint(2,(N,1)).to(torch.float)

        
        net = Fall_Detection_LSTM(dim, args.embed_size, args.lstm_hidden_size, args.lstm_num_layers)
        net.load_state_dict(torch.load(args.model_save_path))
        net.eval()
        with torch.no_grad():
            traj_features = net(traj_batch)
            fall_predict = 1.*(traj_features > 0.5)
            acc = torch.mean(1.*(fall_predict == traj_labels))
            loss = bce_loss_func(traj_features, traj_labels)
            print(f"Evaluation loss = {loss}, accuracy = {acc}")