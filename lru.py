# class LRU(nn.Module):
    def __init__(
            self,
            d_input,
            i_layer = None,# Only needs to be passed into certain residuals like Decay
            prenorm=True,
            dropout=0.0,
            tie_dropout=False,
            transposed=False,
            residual=None, # Config for residual function
            norm=None, # Config for normalization layer
            drop_path=0.,
            **kernel_args,
        ):
        super().__init__()

        self.d_input = d_input
        self.prenorm = prenorm
        self.bidirectional = bidirectional
        self.transposed = transposed

        self.kernel = LRUKernel(self.d_input, **kernel_args)
        # Residual
        # d_residual is the output dimension after residual
        if residual is None:
            self.residual = None
            self.d_residual = self.kernel.d_output
        else:
            self.residual = utils.instantiate(residual_registry, residual, i_layer, d_input, self.kernel.d_output)
            self.d_residual = self.residual.d_output

        # Normalization
        d_norm = d_input if self.prenorm else self.d_residual
        # We don't use config to directly instantiate since Normalization has some special cases
        if norm is None:
            self.norm = None
        elif isinstance(norm, str):
            self.norm = Normalization(d_norm, transposed=self.transposed, _name_=norm)
        else:
            self.norm = Normalization(d_norm, transposed=self.transposed, **norm)

        # Dropout
        dropout_cls = partial(DropoutNd, transposed=self.transposed) if tie_dropout else nn.Dropout
        self.drop = dropout_cls(dropout) if dropout > 0.0 else nn.Identity()
        # self.drop = nn.Dropout1d(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.d_input, 2*self.d_input, kernel_size=1),
            nn.GLU(dim=-2),
        )

        # Stochastic depth
        # self.drop_path = StochasticDepth(drop_path, mode='row') if drop_path > 0.0 else nn.Identity()

        self.activation = nn.GELU()

    def forward(self, x, state=None, **kwargs):
        y = x #(B, L, H) 
        # wandb.log({"y_after/begin": torch.mean(y)})

        # Pre-norm
        if self.norm is not None and self.prenorm: y = self.norm(y) #(B, L, H) 
        
        # Black box layer
        y, _ = self.kernel(y, state=state, **kwargs) #(B, L, H) 
        # print(f"After RNN layer: {torch.mean(y_for)}")
        # wandb.log({"y_after/rnn": torch.mean(y_for), "y_after/rnn_last": torch.mean(new_state)})


        # Post-norm
        if self.norm is not None and not self.prenorm: y = self.norm(y)
        # wandb.log({"y_after/post-norm": torch.mean(y)})   

       ##! Dropout + GELU
        y = self.drop(self.activation(y)) #(B, L, H) 
        # print(f"After activation: {torch.mean(y)}")
        # wandb.log({"y_after/act": torch.mean(y)})

        #! position-wise FFN (GLU)
        if not self.transposed: y = y.transpose(-1, -2) #(B, H, L) 
        y = self.output_linear(y)  
        if not self.transposed: y = y.transpose(-1, -2) #(B, L, H) 
        
        ##! Droput
        y = self.drop(y)
        
        return y + x, state # skip connection