import argparse

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train or eval.')
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--info_per_epoch', type=int, default=5)
    parser.add_argument('--motion_dim', type=int, default=10)
    parser.add_argument('--obs_seq_len', type=int, default=12)
    parser.add_argument('--embed_size', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lstm_hidden_size', type=int, default=32)
    parser.add_argument('--lstm_num_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lr_scheduler', type=str, default='StepLR')    
    parser.add_argument('--lr_scheduler_gamma', type=float, default=0.9)
    # parser.add_argument('--data_file_path', type=str, default='data/imu.txt')    
    parser.add_argument('--data_file_path', type=str, default='data/imu_fall.txt')        
    parser.add_argument('--model_save_path', type=str, default='trained_model/fall_detection_lstm.pt')    
    return parser.parse_args()

    




