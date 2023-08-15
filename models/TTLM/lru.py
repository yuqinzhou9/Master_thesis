import math
import torch
import torch.nn as nn
import torch.nn.functional as F




class LRU(nn.Module):
    """
    LRU module in charge of the recurrent processing.
    Implementation following the one of Orvieto et al. 2023.

    """

    def __init__(self, 
        d_input,
        d_hidden = 128,
        lr = 3e-7, 
        rmin=0, 
        rmax=1,
        max_phase=6.283,
        dropout = 0,
        transposed = True):


        super().__init__()

        # d_model: input and output size
        self.d_input = d_input
        self.d_output= d_input

        # hidden states
        self.d_hidden = d_hidden
        self.transposed = transposed
        

        # lambda 
        ## nu
        u1=torch.rand(self.d_hidden)
        self.nu_log= nn.Parameter(torch.log(-0.5*torch.log(u1*(rmax+rmin)*(rmax-rmin) + rmin**2)))
        self.register("nu_log", self.nu_log, lr)

        ## theta
        u2=torch.rand(self.d_hidden)
        self.theta_log= nn.Parameter(torch.log(max_phase*u2))
        self.register("theta_log", self.theta_log, lr)

        Lambda_mod=torch.exp(-torch.exp(self.nu_log))



        # gamma
        self.gamma_log=nn.Parameter(torch.log(torch.sqrt(torch.ones_like(Lambda_mod)-torch.square(Lambda_mod))))
        self.register("gamma_log", self.gamma_log, lr)

        # B
        B_re=torch.randn([self.d_hidden,self.d_input])/math.sqrt(2*self.d_input)
        B_im=torch.randn([self.d_hidden,self.d_input])/math.sqrt(2*self.d_input)
        self.B=nn.Parameter(torch.complex(B_re,B_im))
        self.register("B", self.B, lr)


        # C
        C_re=torch.randn([self.d_output,self.d_hidden])/math.sqrt(self.d_hidden)
        C_im=torch.randn([self.d_output,self.d_hidden])/math.sqrt(self.d_hidden)
        self.C=nn.Parameter(torch.complex(C_re,C_im))
        self.register("C", self.C, lr)

        # h 
        self.state=torch.complex(torch.zeros(self.d_hidden),torch.zeros(self.d_hidden))

        # D
        self.D=nn.Parameter(torch.randn([self.d_output, self.d_input])/math.sqrt(self.d_input))
        self.register("D", self.D, lr)

    def forward(self, input, state=None, **kwargs):
        """
        (B, L, H) 
        """

        # Construct initial state

        self.state=self.state.to(self.B.device) if state==None else state

        Lambda_mod=torch.exp(-torch.exp(self.nu_log))
        Lambda_re=Lambda_mod*torch.cos(torch.exp(self.theta_log))
        Lambda_im=Lambda_mod*torch.sin(torch.exp(self.theta_log))
        Lambda=torch.complex(Lambda_re,Lambda_im)
        Lambda=Lambda.to(self.state.device)


        gammas=torch.exp(self.gamma_log).unsqueeze(-1).to(self.B.device)
        gammas=gammas.to(self.state.device)
        output=torch.empty([i for i in input.shape[:-1]] +[self.d_output],device=self.B.device)



        #Handle input of (B, L, H)
        for i,batch in enumerate(input):
            out_seq=torch.empty(input.shape[1],self.d_output)
            
            for j,step in enumerate(batch):
                self.state=(Lambda@self.state + gammas* self.B@step.to(dtype= self.B.dtype))
                out_step= (self.C@self.state).real + self.D@step
                out_seq[j]=out_step
            
            self.state=torch.complex(torch.zeros_like(self.state.real),torch.zeros_like(self.state.real))
            output[i]=out_seq
        
        return output, None # Return a dummy state to satisfy this repo's interface, but this can be modified

        
    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)

    def default_state(self, *batch_shape, device=None):
        return torch.randn(
            *batch_shape, self.d_model,
            device=device,
            requires_grad=False
        )



