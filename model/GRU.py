import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.4):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = nn.Dropout(dropout_rate)
        
        # weights
        self.W_u = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(hidden_size, hidden_size + input_size)))
        self.b_u = nn.Parameter(torch.zeros(hidden_size, 1))
        
        self.W_r = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(hidden_size, hidden_size + input_size)))
        self.b_r = nn.Parameter(torch.zeros(hidden_size, 1))
        
        self.W_c = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(hidden_size, hidden_size + input_size)))
        self.b_c = nn.Parameter(torch.zeros(hidden_size, 1))
        
        # Output layer
        self.W_y = nn.Parameter(torch.randn(output_size, hidden_size) * 0.01)
        self.b_y = nn.Parameter(torch.zeros(output_size, 1))
    
    def forward(self, X):
        """Forward function for LSTM across all time step input vector, return output and hidden state for BiLSTM
        
        Args:
            X               (seq_len, input_size, 1): Input sequence data

        Returns:
            outputs         (seq_len, output_size, 1): Output data
            hidden_states   (seq_len, hidden_size, 1): Hidden state across time step
            
        """

        h_t = torch.zeros(self.hidden_size, 1, device=X.device)        
        outputs = []
        hidden_states = []
                
        for x_t in X:
            concat = torch.cat((h_t, x_t), dim=0)
            u_t = torch.sigmoid(torch.mm(self.W_u, concat) + self.b_u)
            r_t = torch.sigmoid(torch.mm(self.W_r, concat) + self.b_r)
            
            h_cand = torch.tanh(torch.mm(self.W_c, torch.cat((r_t * h_t, x_t), dim=0)) + self.b_c)
            h_t = u_t*h_cand + (1 - u_t)*h_t 
            h_t = self.dropout(h_t)           
            hidden_states.append(h_t)

            y_t = torch.mm(self.W_y, h_t) + self.b_y
            outputs.append(y_t)
            
        outputs = torch.stack(outputs, dim=0)
        hidden_states = torch.stack(hidden_states, dim=0)

        return outputs, hidden_states
            