"""
Highway layers
"""

import torch
from torch import nn


class Highway(nn.Module):
    """
    Individual highway layer
    """

    def __init__(self, input_dim, activation_class=nn.ReLU):
        """
        Create a highway layer. The input is a tensor of features, the output
        is a tensor with the same dimension.

        With input $x$, return $y$:
            $g = \sigma(W_gx+b_g)$
            $n = f(W_nx+b_n)$
            $y = gx + (1-g)n$

        Parameters:
            :param: input_dim (int): the input dimensionality
            :param: activation_class (nn.Module): the class of the
            non-linearity. Default: ReLU

        Input:
            :param: input: a float tensor with shape [batch, input_dim]

        Return:
            :return: a float tensor with shape [batch, input_dim]
        """
        super(Highway, self).__init__()
        self.input_dim = input_dim
        self.layer = nn.Linear(input_dim, input_dim*2)
        self.activation = activation_class()
        self.gate = nn.Sigmoid()

        # We want untrained layers to copy their input.
        # To do so, the bias is initialized with one.
        self.layer.bias[input_dim:].data.fill_(1)
        return

    def forward(self, input):
        projected = self.layer(input)
        non_lin, gate = torch.split(projected, self.input_dim, -1)
        non_lin = self.activation(non_lin)
        gate = self.gate(gate)
        combined = gate * input + (1 - gate) * non_lin
        return combined


class Highways(nn.Module):
    """
    Stack of highway layers. Convenience class.
    """

    def __init__(self, input_dim, n_layers, activation_class=nn.ReLU):
        """
        Create n_layers highway layers. The input is a tensor of features,
        the output is a tensor with the same dimension.

        Parameters:
            :param: input_dim (int): the input dimensionality
            :param: n_layers (int): the number of layers
            :param: activation_class (nn.Module): the class of the
            non-linearity. Default: ReLU

        Sub-modules:
            layers: the stack of highway layers

        Input:
            :param: input: a float tensor with shape [batch, input_dim]

        Return:
            :return: a float tensor with shape [batch, input_dim]
        """
        super(Highways, self).__init__()
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.layers = nn.Sequential(*[
            Highway(input_dim, activation_class)
            for _ in range(n_layers)])
        return

    def forward(self, input):
        return self.layers(input)
