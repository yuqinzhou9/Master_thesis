import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from operator import attrgetter



class TTLM(nn.Module):
    def __init__(self, 
        d_input,
        d_hidden = 128,
        lr = 3e-7, 
        rmin=0, 
        rmax=1,
        max_phase=6.283,
        dropout = 0,
        return_output=True,
        transposed = True):


        super().__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ## register weight decay and the factored lr

        self.return_output = return_output
        # d_model: input and output size
        self.d_input = d_input
        self.d_output= d_input
        self.lr = lr

        # hidden states
        self.d_hidden = d_hidden
        self.transposed = transposed
        self.reset_parameters()
        # optim = {"weight_decay": 0.0, "lr": lr}
        # for para in self.named_parameters():
        #     setattr(attrgetter(para[0])(self), "_optim", optim)
        

    def forward(self, inputs, state=None, **kwargs):
        """
        Input (B, L, H) 
        """
        inputs = inputs.transpose(0, -1) #(H, L, B) 

        # Construct initial state
        # state =  torch.randn(self.d_hidden, inputs.shape[-1], device = inputs.device, requires_grad=False) # State: (H, B)
        state =  self.default_state(inputs.shape[-1]).to(self.device) # State: (H, B)
        outputs = []
        for input in torch.unbind(inputs, dim=-2): 
            output, new_state = self.step(input, state) ## Inputs: (N, B) State: # Inputs: (H, B)
            state = new_state
            if self.return_output:
                outputs.append(output)
            
        outputs = torch.stack(outputs, dim=-2) if self.return_output else None ## (H, L, B)
        outputs = outputs.transpose(0, -1) #(B, L, H) 
        return outputs, state
    

    def step(self, input, state):
        state_preac = self.A @ state + self.B @ input
        return state, state


    def reset_parameters(self):
        self.B = nn.Parameter(nn.init.xavier_normal_(torch.empty(self.d_hidden, self.d_input)).to(self.device))
        self.register("B", self.B, self.lr)
        self.reset_hidden_to_hidden()


    def reset_hidden_to_hidden(self):
        self.A = nn.Parameter(nn.init.xavier_normal_(torch.empty(self.d_hidden, self.d_hidden)).to(self.device))
        self.register("A", self.A, self.lr)


    def default_state(self, batch_shape, device=None):
        return torch.zeros(self.d_hidden, batch_shape, requires_grad=False)

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)