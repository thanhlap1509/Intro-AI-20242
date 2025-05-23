import torch
import torch.nn as nn
from LSTM import LSTM

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.4):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.LSTM_cell_l = LSTM(input_size, hidden_size, output_size, dropout_rate)
        self.LSTM_cell_r = LSTM(input_size, hidden_size, output_size, dropout_rate)
        
        self.W_y = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(output_size, hidden_size * 2)))
        self.b_y = nn.Parameter(torch.zeros(output_size, 1))
        
    def forward(self, X):
        """Fowarding X input sequence across two LSTM sequence, left-right and right-left

        Args:
            X       (seq_len, input_size, 1)       : Input sequence data
        Data:
            h_left  (seq_len, hidden_state, 1)  : Hidden states along the beginning-end time step
            h_right (seq_len, hidden_state, 1)  : Hidden states along the end-beginning time step
            h       (seq_len, hidden_state*2, 1): Concatenation of hidden state from LSTM left and LSTM right in a beginning-end fashion
        Returns:
            outputs (seq_len, output_size, 1)   : Output data
            h       (seq_len, hidden_state*2, 1): Concatenation of hidden state from LSTM left and LSTM right in a beginning-end fashion

        """
        _, hs_left = self.LSTM_cell_l(X)
        _, hs_right = self.LSTM_cell_r(torch.flip(X,dims=[0])) # flip input along time dimension for right LSTM
        hs_right = torch.flip(hs_right, dims=[0]) # flip right hidden state to be from beginning-end
                        
        h = torch.cat((hs_left, hs_right),dim=1)
        
        outputs = torch.matmul(self.W_y, h.squeeze(-1).T) + self.b_y  # shape: (output_size, seq_len)
        outputs = outputs.T.unsqueeze(-1)  # shape: (seq_len, output_size, 1)
          
        return outputs, h

if __name__ == "__main__":
    model = BiLSTM(13, 200, 1)
    x = torch.randn(100, 13, 1)
    print(model(x)[0].shape)  
        