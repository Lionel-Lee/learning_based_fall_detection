import torch
import torch.nn as nn

class Fall_Detection_LSTM(nn.Module):
    def __init__(
        self,
        motion_dim,
        embed_size,
        lstm_hidden_size,
        lstm_num_layers,
    ):
        super(Fall_Detection_LSTM, self).__init__()
        self.motion_dim = motion_dim
        self.embed_size = embed_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers

        self.traj_node_embed = nn.Linear(motion_dim, embed_size)
        self.traj_model = nn.LSTM(
            input_size = embed_size,
            hidden_size = lstm_hidden_size,
            num_layers = lstm_num_layers,
            batch_first=False,
            dropout=0.,
            bidirectional=False,
        )
        self.head = nn.Linear(
            lstm_num_layers*lstm_hidden_size,
            2,
        )

    def forward(
        self,
        b_traj_inputs,
    ):
        """
        inputs:
            - b_traj_inputs: trajectory batch.
                # Raw features processed from global position sequences of all 
                # considered agents with dimensions (N, obs_seq_len, motion_dim)
        outputs:
            - features: processed agent traj features for binary classification
                # (N, 2)
        """
        features = self._lstm_forward(b_traj_inputs)
        features = self.head(features)
        return torch.sigmoid(features)
        

    def _lstm_forward(
        self,
        b_traj_inputs,
    ):
        """
        inputs:
            - b_traj_inputs: trajectory batch.
                # (N, obs_seq_len, motion_dim)
        outputs:
            - h_osl: processed agent node features
                # (N, lstm_num_layers*lstm_hidden_size)
                # osl stands for observation sequence length.
        """
        b_size, obs_seq_len, _ = b_traj_inputs.shape
        b_traj_inputs = b_traj_inputs.permute([1,0,2])  # (obs_seq_len, N, motion_dim)
        b_traj_inputs = self.traj_node_embed(b_traj_inputs)
        _, (h_osl, c_osl) = self.traj_model(b_traj_inputs) # (lstm_n_layers, N, lstm_hidden_size)
        h_osl = h_osl.permute([1,0,2]) # (N, lstm_n_layers, lstm_hidden_size)
        h_osl = h_osl.reshape([b_size, -1])
        return h_osl
