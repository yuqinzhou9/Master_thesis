"""Baseline simple RNN cells such as the vanilla RNN and GRU."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.nn import LinearActivation, Activation # , get_initializer
from src.models.nn.gate import Gate
from src.models.nn.orthogonal import OrthogonalLinear
from src.models.sequence.base import SequenceModule
from operator import attrgetter
import wandb

class CellBase(SequenceModule):
    """Abstract class for our recurrent cell interface.

    Passes input through.
    """
    registry = {}

    # https://www.python.org/dev/peps/pep-0487/#subclass-registration
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Only register classes with @name attribute
        if hasattr(cls, 'name') and cls.name is not None:
            cls.registry[cls.name] = cls

    name = 'id'
    valid_keys = []

    @property
    def default_initializers(self):
        return {}

    @property
    def default_architecture(self):
        return {}

    def __init__(self, d_input, d_model, lr, initializers=None, architecture=None):
        super().__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.lr = lr

        self.architecture = self.default_architecture
        self.initializers = self.default_initializers
        if initializers is not None:
            self.initializers.update(initializers)
            print("Initializers:", initializers)
        if architecture is not None:
            self.architecture.update(architecture)

        assert set(self.initializers.keys()).issubset(self.valid_keys)
        assert set(self.architecture.keys()).issubset(self.valid_keys)

        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, input, hidden):
        """Returns output, next_state."""
        return input, input

    def default_state(self, *batch_shape, device=None):
        return torch.zeros(
            *batch_shape, self.d_model,
            device=device,
            requires_grad=False,
        )

    def step(self, x, state):
        return self.forward(x, state)

    @property
    def state_to_tensor(self):
        return lambda state: state

    @property
    def d_state(self):
        return self.d_model

    @property
    def d_output(self):
        return self.d_model


class RNNCell(CellBase):
    name = 'rnn'

    valid_keys = ['hx', 'hh', 'bias']

    default_initializers = {
        'hx': 'xavier',
        'hh': 'xavier',
    }

    default_architecture = {
        'bias': False,   ###! no bias
    }


    def __init__(
            self, d_input, d_model, lr,
            hidden_activation='tanh',
            orthogonal=False,
            ortho_args=None,
            zero_bias_init=False,
            **kwargs
        ):

        self.hidden_activation = hidden_activation
        self.orthogonal = orthogonal
        self.ortho_args = ortho_args
        self.zero_bias_init=zero_bias_init


        super().__init__(d_input, d_model, lr, **kwargs)
        ## register weight decay and the factored lr
        optim = {"weight_decay": 0.0, "lr": lr}
        for para in self.named_parameters():
            setattr(attrgetter(para[0])(self), "_optim", optim)

    def reset_parameters(self):
        self.W_hx = LinearActivation(
            self.d_input, self.d_model,
            bias=self.architecture['bias'],
            zero_bias_init=self.zero_bias_init,
            initializer=self.initializers['hx'],
            activation=self.hidden_activation,
            # apply_activation=False,
            activate=False,
        )
        self.activate = Activation(self.hidden_activation, self.d_model)
        self.reset_hidden_to_hidden()


    def reset_hidden_to_hidden(self):

        if self.orthogonal:

            if self.ortho_args is None:
                self.ortho_args = {}
            self.ortho_args['d_input'] = self.d_model
            self.ortho_args['d_output'] = self.d_model

            self.W_hh = OrthogonalLinear(**self.ortho_args)
        else:
            self.W_hh = LinearActivation(
                self.d_model, self.d_model,
                bias=self.architecture['bias'],
                zero_bias_init=self.zero_bias_init,
                initializer=self.initializers['hh'],
                activation=self.hidden_activation,
                # apply_activation=False,
                activate=False,
            )
            # self.W_hh = nn.Linear(self.d_model, self.d_model, bias=self.architecture['bias'])
            # get_initializer(self.initializers['hh'], self.hidden_activation)(self.W_hh.weight)




    def forward(self, input, h):
        # Update hidden state
        # print(f"h: {torch.mean(h)}; W^hh: {torch.mean(self.W_hh.weight.data)}; W^hh@h: {torch.mean(self.W_hh(h))}")
        # wandb.log({"h_after/rnn": torch.mean(h)})
        hidden_preact = self.W_hx(input) + self.W_hh(h)
        hidden = self.activate(hidden_preact)
        ###! the last hidden state
        return hidden, hidden


class MIRNNCell(CellBase):
    name = 'mirnn'

    valid_keys = ['hx', 'hh', 'bias']

    default_initializers = {
        'hx': 'xavier',
        'hh': 'xavier',
    }

    default_architecture = {
        'bias': False,   ###! no bias
    }


    def __init__(
            self, d_input, d_model, lr,
            hidden_activation='tanh',
            orthogonal=False,
            ortho_args=None,
            zero_bias_init=False,
            **kwargs
        ):

        self.hidden_activation = hidden_activation
        self.orthogonal = orthogonal
        self.ortho_args = ortho_args
        self.zero_bias_init=zero_bias_init


        super().__init__(d_input, d_model, lr, **kwargs)
        ## register weight decay and the factored lr
        optim = {"weight_decay": 0.0, "lr": lr}
        for para in self.named_parameters():
            setattr(attrgetter(para[0])(self), "_optim", optim)

    def reset_parameters(self):
        self.W_hx = LinearActivation(
            self.d_input, self.d_model,
            bias=self.architecture['bias'],
            zero_bias_init=self.zero_bias_init,
            initializer=self.initializers['hx'],
            activation=self.hidden_activation,
            # apply_activation=False,
            activate=False,
        )
        self.activate = Activation(self.hidden_activation, self.d_model)
        self.reset_hidden_to_hidden()


    def reset_hidden_to_hidden(self):

        if self.orthogonal:

            if self.ortho_args is None:
                self.ortho_args = {}
            self.ortho_args['d_input'] = self.d_model
            self.ortho_args['d_output'] = self.d_model

            self.W_hh = OrthogonalLinear(**self.ortho_args)
        else:
            self.W_hh = LinearActivation(
                self.d_model, self.d_model,
                bias=self.architecture['bias'],
                zero_bias_init=self.zero_bias_init,
                initializer=self.initializers['hh'],
                activation=self.hidden_activation,
                # apply_activation=False,
                activate=False,
            )
            # self.W_hh = nn.Linear(self.d_model, self.d_model, bias=self.architecture['bias'])
            # get_initializer(self.initializers['hh'], self.hidden_activation)(self.W_hh.weight)


    def default_state(self, *batch_shape, device=None):
        return torch.randn(
            *batch_shape, self.d_model,
            device=device,
            requires_grad=False
        )
        
        # return nn.init.xavier_normal_(torch.empty(
        #     *batch_shape, self.d_model,
        #     device=device,
        #     requires_grad=False)
        # )

        # return nn.init.uniform_(torch.empty(
        #     *batch_shape, self.d_model,
        #     device=device,
        #     requires_grad=False,
        # ), 0.9, 0.99)

    def forward(self, input, h):
        # Update hidden state
        # wandb.log({"h_after/rnn": torch.mean(h)})
        hidden_preact = self.W_hx(input) * self.W_hh(h)
        # print(f"h: {torch.mean(h)}; W^hh: {torch.mean(self.W_hh.weight.data)}; W^hh@h: {torch.mean(self.W_hh(h))}")
        hidden = self.activate(hidden_preact)
        ###! the last hidden state
        return hidden, hidden



class TTLM(CellBase):
    name = 'ttlm'
    def __init__(
            self, d_input, d_model, lr,
            **kwargs
        ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        super().__init__(d_input, d_model, lr, **kwargs)
        ## register weight decay and the factored lr
        optim = {"weight_decay": 0.0, "lr": lr}
        for para in self.named_parameters():
            setattr(attrgetter(para[0])(self), "_optim", optim)

        self.activation = nn.Tanh()
    


    def forward(self, input, h):
        hidden = self.activation((self.A @ h.to(dtype= self.A.dtype)) * (self.B @ input.to(dtype= self.B.dtype)))
        # hidden = torch.log(self.A @ h.to(dtype= self.A.dtype)) + torch.log(self.B @ input.to(dtype= self.B.dtype))
        hidden_out = (self.C @ hidden).real + self.D @ input
        print(torch.mean(hidden))
        # wandb.log({"after/hidden_out": torch.mean(hidden_out), "after/hidden": torch.mean(hidden.real)})
        return hidden_out, hidden_out

    def default_state(self, *batch_shape, device=None):
        return torch.complex(torch.randn(self.d_model, *batch_shape, device=device,requires_grad=False),
        torch.randn(self.d_model, *batch_shape,device=device,requires_grad=False))

    def reset_parameters(self):
        C_re = nn.init.xavier_normal_(torch.empty(self.d_model, self.d_model))
        C_im = nn.init.xavier_normal_(torch.empty(self.d_model, self.d_model))
        self.C = nn.Parameter(torch.complex(C_re,C_im).to(self.device))
        self.D = nn.Parameter(nn.init.xavier_normal_(torch.empty(self.d_model, self.d_input)).to(self.device))

        self.reset_hidden_to_hidden()
        self.reset_hidden_to_input()


    def reset_hidden_to_hidden(self):
        A_re = nn.init.xavier_normal_(torch.empty(self.d_model, self.d_model))
        A_im = nn.init.xavier_normal_(torch.empty(self.d_model, self.d_model))
        self.A = nn.Parameter(torch.complex(A_re,A_im).to(self.device))

    def reset_hidden_to_input(self):
        B_re = nn.init.xavier_normal_(torch.empty(self.d_input, self.d_model))
        B_im = nn.init.xavier_normal_(torch.empty(self.d_input, self.d_model))
        self.B = nn.Parameter(torch.complex(B_re,B_im).to(self.device))