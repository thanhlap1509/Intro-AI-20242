import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # weights
        self.W_f = nn.Parameter(torch.randn(hidden_size, hidden_size + input_size))
        self.b_f = nn.Parameter(torch.randn(hidden_size, 1))
        
        self.W_i = nn.Parameter(torch.randn(hidden_size, hidden_size + input_size))
        self.b_i = nn.Parameter(torch.randn(hidden_size, 1))
        
        self.W_c = nn.Parameter(torch.randn(hidden_size, hidden_size + input_size))
        self.b_c = nn.Parameter(torch.randn(hidden_size, 1))
        
        self.W_o = nn.Parameter(torch.randn(hidden_size, hidden_size + input_size))
        self.b_o = nn.Parameter(torch.randn(hidden_size, 1))
        
        # Output layer
        self.W_y = nn.Parameter(torch.randn(output_size, hidden_size))
        self.b_y = nn.Parameter(torch.zeros(output_size, 1))
    
    def forward(self, X):
        
        """
            X: (sequence len, input size) Input sequence data
            outputs: (sequence len, output_size) - Output data
        """
        h_t = torch.randn(self.hidden_size, 1)
        c_t = torch.randn(self.hidden_size, 1)
        
        outputs = []
                
        for x_t in X:
            f_t = torch.sigmoid(torch.mm(self.W_f, torch.cat((h_t, x_t), dim=0)) + self.b_f)
            i_t = torch.sigmoid(torch.mm(self.W_i, torch.cat((h_t, x_t), dim=0)) + self.b_i)
            c_cand = torch.tanh(torch.mm(self.W_c, torch.cat((h_t, x_t), dim=0)) + self.b_c)
            
            c_t = f_t*c_t + i_t*c_cand
            
            o_t = torch.sigmoid(torch.mm(self.W_o, torch.cat((h_t, x_t), dim=0)) + self.b_o)
            h_t = o_t*torch.tanh(c_t)
            
            y_t = torch.mm(self.W_y, h_t) + self.b_y
            outputs.append(y_t.unsqueeze(0))
            
        return outputs
            