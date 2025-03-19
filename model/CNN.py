import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, kernel_size, num_filters, kernel_stride, pool_size, pool_stride):
        super().__init__()
        
        self.filters = torch.nn.Conv1d(in_channels = 1, 
                                       out_channels = num_filters, 
                                       kernel_size = kernel_size, 
                                       stride = kernel_stride,
                                       padding = kernel_size // 2) # for same size padding, require kernel_size odd
        
        self.max_pool = nn.MaxPool1d(kernel_size=pool_size, 
                                     stride=pool_stride, 
                                     padding = pool_size // 2) # for same padding, require pool_size odd
        
    def forward(self, X):
    
        """Forwarding function for CNN model
        
        Args:
            X       (seq_len, input_size): Input sequence data, seq_len is number of time step, input_size is size of feature vector
            
        Returns:
            X       (seq_len, num_filters * L_out): higher order representation of input data
        
        """
        X = X.unsqueeze(dim=1)  # X = (seq_len, 1, input_size) = (N, C_in, L) for convolution 
        X = self.filters(X)     # X = (seq_len, num_filters, L_out)
        X = self.max_pool(X)    # X = (seq_len, num_filters, l_pooled)
        X = torch.flatten(X, start_dim = 1) # X = (seq_len, num_filters * L_pooled) concat hidden feature vector from different filters to form a singular vector per time step
        return X