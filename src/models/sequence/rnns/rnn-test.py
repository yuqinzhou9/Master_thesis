import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.sequence.rnns.rnn import RNN



class RNNbased(nn.Module):

    def __init__(
        self,
        d_model=256,
        n_layers=2,
        dropout=0.2,
        prenorm=True,
        transposed=True,
        cell=None
    ):
        super().__init__()

        self.d_model = d_model
        self.d_output = self.d_model #S4-package requirements
        self.prenorm = prenorm
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.FFNs = nn.ModuleList()

        for _ in range(n_layers):
            self.layers.append(
                RNN(d_model = self.d_model, cell = cell, return_output=True, transposed=True, dropout=0)
            )
            # self.norms.append(nn.LayerNorm(d_model))
            self.norms.append(nn.BatchNorm1d(d_model)) 
            self.dropouts.append(nn.Dropout1d(dropout))
            self.FFNs.append(nn.Sequential(nn.Conv1d(d_model, d_model*2, kernel_size= 1 ), nn.GLU(dim=-2))  #                
                                 )

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        

        for layer, norm, dropout, FFN in zip(self.layers, self.norms, self.dropouts, self.FFNs):
            ''' Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L) '''

            
            z = x #(B, d_model, L) -> (B, d_model, L) 
            if self.prenorm:
                # Prenorm (BN)
                z = norm(z) # (B, d_model, L) -> (B, d_model, L) 

            # Apply recurrence: we ignore the state input and output
            z, _ = layer(z) #(B, d_model, L) -> (B, d_model, L) (note that we transpose the input inside the layer)

            
            # Dropout on the output of the MLP 
            z = dropout(z) #(B, d_model, L) -> (B, d_model, L) for dropout1d
            
            # MLP +GLP
            z = FFN(z) #(B, d_model, L) -> (B, d_model, L) for conv1d

            # Residual connection
            x = z + x  #(B, d_model, L) -> (B, d_model, L)

            if not self.prenorm:
                # Post-norm (BN)
                x = norm(x) #(B, d_model, L) -> (B, d_model, L)
                
        # Pooling: average pooling over the sequence length
        x = x.transpose(-1, -2)
        x = x.mean(dim=1) # (B, L, d_model) -> (B, d_model)
        return x