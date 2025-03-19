import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate = 0.4):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = nn.Dropout(dropout_rate)
        
        # weights
        self.W_f = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(hidden_size, hidden_size + input_size)))
        self.b_f = nn.Parameter(torch.zeros(hidden_size, 1))
        
        self.W_i = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(hidden_size, hidden_size + input_size)))
        self.b_i = nn.Parameter(torch.zeros(hidden_size, 1))
        
        self.W_c = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(hidden_size, hidden_size + input_size)))
        self.b_c = nn.Parameter(torch.zeros(hidden_size, 1))
        
        self.W_o = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(hidden_size, hidden_size + input_size)))
        self.b_o = nn.Parameter(torch.zeros(hidden_size, 1))
        
        # Output layer
        self.W_y = nn.Parameter(torch.randn(output_size, hidden_size) * 0.01)
        self.b_y = nn.Parameter(torch.zeros(output_size, 1))
    
    def forward(self, X):
        """Forward function for LSTM across all time step input vector, return output and hidden state for BiLSTM
        
        Args:
            X               (seq_len, input_size): Input sequence data

        Returns:
            outputs         (seq_len, output_size, 1): Output data
            hidden_states   (seq_len, hidden_size, 1): Hidden state across time step
            
        """

        h_t = torch.zeros(self.hidden_size, 1, device=X.device)
        c_t = torch.zeros(self.hidden_size, 1, device=X.device)
        
        outputs = []
        hidden_states = []
                
        for x_t in X:
            concat = torch.cat((h_t, x_t), dim=0)
            
            f_t = torch.sigmoid(torch.mm(self.W_f, concat) + self.b_f)
            i_t = torch.sigmoid(torch.mm(self.W_i, concat) + self.b_i)
            c_cand = torch.tanh(torch.mm(self.W_c, concat) + self.b_c)
            
            c_t = f_t*c_t + i_t*c_cand
            
            o_t = torch.sigmoid(torch.mm(self.W_o, concat) + self.b_o)
            h_t = o_t*torch.tanh(c_t)
            h_t = self.dropout(h_t)
            
            y_t = torch.mm(self.W_y, h_t) + self.b_y
            hidden_states.append(h_t)
            outputs.append(y_t)
            
        outputs = torch.stack(outputs, dim=0)
        hidden_states = torch.stack(hidden_states, dim=0)

        return outputs, hidden_states
            