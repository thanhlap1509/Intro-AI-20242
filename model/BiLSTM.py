import torch
import torch.nn as nn
from LSTM import LSTM

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.LSTM_cell_l = LSTM(input_size, hidden_size, output_size)
        self.LSTM_cell_r = LSTM(input_size, hidden_size, output_size)
        
        self.W_y = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(output_size, hidden_size * 2)))
        self.b_y = nn.Parameter(torch.zeros(output_size, 1))
        
    def forward(self, X):
        """Fowarding X input sequence across two LSTM sequence, left-right and right-left

        Args:
            X       (seq_len, input_size)       : Input sequence data
        Data:
            h_left  (seq_len, hidden_state, 1)  : Hidden states along the beginning-end time step
            h_right (seq_len, hidden_state, 1)  : Hidden states along the end-beginning time step
            h       (seq_len, hidden_state*2, 1): Concatenation of hidden state from LSTM left and LSTM right in a beginning-end fashion
        Returns:
            outputs (seq_len, output_size, 1)   : Output data
        """
        _, hs_left = self.LSTM_cell_l(X)
        _, hs_right = self.LSTM_cell_r(torch.flip(X,dims=[0])) # flip input along time dimension for right LSTM
        hs_right = torch.flip(hs_right, dims=[0]) # flip right hidden state to be from beginning-end
                
        outputs = []
        
        h = torch.cat((hs_left, hs_right),dim=1)
        
        for h_t in h:
            y_t = torch.mm(self.W_y, h_t) + self.b_y
            outputs.append(y_t)
        
        outputs = torch.stack(outputs, dim=0)
          
        return outputs
        
        