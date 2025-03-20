import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_size, kernel_size, num_filters, pool_size, kernel_stride = 1, pool_stride = 1):
        super().__init__()
        
        self.input_size = input_size
        self.filters = torch.nn.Conv1d(in_channels = 1, 
                                       out_channels = num_filters, 
                                       kernel_size = kernel_size, 
                                       stride = kernel_stride,
                                       padding = kernel_size // 2) # for same size padding, require kernel_size odd and stride = 1
        
        self.max_pool = nn.MaxPool1d(kernel_size = pool_size, 
                                     stride = pool_stride, 
                                     padding = 0) 
        
        # Since we apply the same padding for convolution op, the output size is: 
        self.output_size = num_filters * (((input_size - pool_size) / pool_stride) + 1) 
        
    def forward(self, X):
    
        """Forwarding function for CNN model
        
        Args:
            X       (seq_len, input_size): Input sequence data, seq_len is number of time step, input_size is size of feature vector
            
        Returns:
            X       (seq_len, num_filters * L_pooled): higher order representation of input data
        
        """
        X = X.unsqueeze(dim=1)  # X = (seq_len, 1, input_size) 
        X = self.filters(X)     # X = (seq_len, num_filters, input_size) = (N, C_in, L_out) 
        X = self.max_pool(X)    # X = (seq_len, num_filters, ((input_size - pool_size) / pool_stride) + 1) 
        X = torch.flatten(X, start_dim = 1) # X = (seq_len, num_filters * ((input_size - pool_size) / pool_stride) + 1)) 
        return X