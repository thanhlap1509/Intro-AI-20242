import torch
import torch.nn as nn
from model import CNN, LSTM

class CNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, kernel_size, num_filters, pool_size, kernel_stride = 1, pool_stride = 1, dropout_rate=0.4):
        super().__init__()
        
        self.cnn = CNN(input_size = input_size,
                       kernel_size = kernel_size,
                       num_filters = num_filters,
                       pool_size = pool_size
                       )
        
        self.lstm = LSTM(self.cnn.output_size, hidden_size, output_size)

def forward(self, X):
    
        """Forwarding function for CNN-LSTM model
        
        Args:
            X       (seq_len, input_size, 1): Input sequence data, seq_len is number of time step, input_size is size of feature vector
            
        Returns:
            X       (seq_len, hidden_size, 1): higher order representation of input data
        
        """
        X = self.cnn(X)
        outputs, hidden_states = self.lstm(X)
        return outputs, hidden_states
        
      