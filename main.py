from utils.args import arg_parse
from model.fall_detection_lstm import Fall_Detection_LSTM
from utils.traj_data_loader import MINI_Traj_Dataset
from torch.utils.data import DataLoader
# from torch.autograd import Variable
import torch
import os
import time
from torch.optim.lr_scheduler import StepLR


import matplotlib.pyplot as plt
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

        epoch_train_loss = []
        epoch_train_acc = []
        epoch_eval_loss = []
        epoch_eval_acc = []
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

            epoch_train_acc.append(avg_acc/num_batch)
            epoch_train_loss.append(avg_loss/num_batch)
            if args.lr_scheduler is not None:
                lr_scheduler.step()

            # eval in training for report
            net.eval()
            eval_traj_dataset = MINI_Traj_Dataset(data_file_path = 'data/imu_eval.txt', obs_seq_len = args.obs_seq_len)
            eval_MINI_Traj_data_loader = DataLoader(dataset=traj_dataset, batch_size = N, shuffle = True, drop_last=True)

            with torch.no_grad():
                eval_avg_loss = 0.
                eval_avg_acc = 0.
                eval_num_batch = 1.*len(eval_MINI_Traj_data_loader)
                for _, eval_batch in enumerate(eval_MINI_Traj_data_loader):
                    #random fake inputs for labels, need dataloader for real imu data
                    eval_traj_batch, eval_traj_labels = eval_batch    
                    eval_traj_batch = eval_traj_batch.to(device)
                    eval_traj_labels = eval_traj_labels.to(device)

                    eval_traj_features = net(eval_traj_batch)
                    eval_fall_predict = 1.*(eval_traj_features > 0.5)
                    eval_acc = torch.mean(1.*(eval_fall_predict == eval_traj_labels))

                    eval_loss = bce_loss_func(eval_traj_features, eval_traj_labels)
                    eval_avg_loss += eval_loss
                    eval_avg_acc += eval_acc
                epoch_eval_acc.append(eval_avg_acc/eval_num_batch)
                epoch_eval_loss.append(eval_avg_loss/eval_num_batch)

        epochs = range(1, args.num_epoch+1)
        plt.plot(epochs, epoch_train_loss, 'g', label='Training loss')
        plt.plot(epochs, epoch_eval_loss, 'b', label='validation loss')
        plt.title('Training and Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig("loss_plot.png")
        plt.close("all")

        plt.plot(epochs, epoch_train_acc, 'g', label='Training accuracy')
        plt.plot(epochs, epoch_eval_acc, 'b', label='validation accuracy')
        plt.title('Training and Validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig("acc_plot.png")
        plt.close("all")

        torch.save(net.state_dict(), args.model_save_path)

    # python main.py --mode eval --data_file_path data/imu_eval.txt
    elif args.mode == 'eval':        
        net = Fall_Detection_LSTM(dim, args.embed_size, args.lstm_hidden_size, args.lstm_num_layers)
        net.load_state_dict(torch.load(args.model_save_path))
        net.to(device)
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
        net = Fall_Detection_LSTM(dim, args.embed_size, args.lstm_hidden_size, args.lstm_num_layers)
        net.load_state_dict(torch.load(args.model_save_path))
        net.to(device)
        net.eval()
        while (True):
            time.sleep(0.02)
            file_name = 'imu_deploy.txt'
            src_path = '/home/mini/' + file_name
            dest_path = '/home/lionel/ECE598JK/learning_based_fall_detection/data/'
            ret = os.system('sshpass -p \"1234\" scp mini@192.168.102.32:'+src_path+' '+dest_path)
            if ret == 0:
                data_file = open((dest_path+file_name), 'r')
                lines = data_file.readlines()
                imu_data_traj = torch.tensor([list(map(float, line.split())) for line in lines][0]).to(device)
                traj_features = net(imu_data_traj)
                if (traj_features > 0.5):
                    print("Fall Detected!")                
