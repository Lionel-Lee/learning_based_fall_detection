import argparse

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train or eval.')
    parser.add_argument('--motion_dim', type=int, default=3)
    parser.add_argument('--obs_seq_len', type=int, default=12)
    parser.add_argument('--embed_size', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lstm_hidden_size', type=int, default=32)
    parser.add_argument('--lstm_num_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lr_scheduler', type=str, default='StepLR')    
    parser.add_argument('--clip_grad', type=float, default=None, help='gradient clipping')
    return parser.parse_args()

    




