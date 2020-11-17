import math
import torch
import torch.nn as nn
import numpy as np

from outside_index import get_outside_index, OutsideIndexCheck
from inside_index import get_inside_index, InsideIndexCheck
from inside_index import get_inside_index_unique
from offset_cache import get_offset_cache

from inside_index import get_inside_components
from outside_index import get_outside_components

from base_model import *

TINY = 1e-8


# Composition Functions

class ComposeMLP(nn.Module):
    def __init__(self, size, activation, n_layers=2, leaf=False, side_1_size=None, side_2_size=None):
        super(ComposeMLP, self).__init__()

        self.size = size
        self.activation = activation
        self.n_layers = n_layers

        if leaf:
            self.V = nn.Parameter(torch.FloatTensor(self.size, self.size))
        self.W = nn.Parameter(torch.FloatTensor(2 * self.size, self.size))
        self.B = nn.Parameter(torch.FloatTensor(self.size))

        self.side_1_size = side_1_size
        if side_1_size is not None:
            self.W_side_1 = nn.Parameter(torch.FloatTensor(side_1_size, self.size))

        self.side_2_size = side_2_size
        if side_2_size is not None:
            self.W_side_2 = nn.Parameter(torch.FloatTensor(side_2_size, self.size))

        for i in range(1, n_layers):
            setattr(self, 'W_{}'.format(i), nn.Parameter(torch.FloatTensor(self.size, self.size)))
            setattr(self, 'B_{}'.format(i), nn.Parameter(torch.FloatTensor(self.size)))
        self.reset_parameters()

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_cuda(self):
        device = self.device
        return device.index is not None and device.index >= 0

    def reset_parameters(self):
        params = [p for p in self.parameters() if p.requires_grad]
        for i, param in enumerate(params):
            param.data.normal_()

    def leaf_transform(self, x, side=None):
        h = torch.tanh(torch.matmul(x, self.V) + self.B)
        device = torch.cuda.current_device() if self.is_cuda else None

        return h

    def forward(self, hs, constant=1.0, side_1=None, side_2=None):
        input_h = torch.cat(hs, 1)
        h = torch.matmul(input_h, self.W)
        if side_1 is not None:
            h = h + torch.matmul(side_1, self.W_side_1)
        if side_2 is not None:
            h = h + torch.matmul(side_2, self.W_side_2)
        h = self.activation(h + self.B)
        for i in range(1, self.n_layers):
            W = getattr(self, 'W_{}'.format(i))
            B = getattr(self, 'B_{}'.format(i))
            h = self.activation(torch.matmul(h, W) + B)

        device = torch.cuda.current_device() if self.is_cuda else None

        return h


# Score Functions

class Bilinear(nn.Module):
    def __init__(self, size_1, size_2=None):
        super(Bilinear, self).__init__()
        self.size_1 = size_1
        self.size_2 = size_2 or size_1
        self.mat = nn.Parameter(torch.FloatTensor(self.size_1, self.size_2))
        self.reset_parameters()

    def reset_parameters(self):
        params = [p for p in self.parameters() if p.requires_grad]
        for i, param in enumerate(params):
            param.data.normal_()

    def forward(self, vector1, vector2):
        # bilinear
        # a = 1 (in a more general bilinear function, a is any positive integer)
        # vector1.shape = (b, m)
        # matrix.shape = (m, n)
        # vector2.shape = (b, n)
        bma = torch.matmul(vector1, self.mat).unsqueeze(1)
        ba = torch.matmul(bma, vector2.unsqueeze(2)).view(-1, 1)
        return ba


# Base

class DioraMLP(DioraBase):
    K = 1

    def __init__(self, *args, **kwargs):
        self.n_layers = kwargs.get('n_layers', None)
        super(DioraMLP, self).__init__(*args, **kwargs)

    @classmethod
    def from_kwargs_dict(cls, context, kwargs_dict):
        return cls(**kwargs_dict)

    def init_parameters(self):
        # Model parameters for transformation required at both input and output
        self.inside_score_func = Bilinear(self.size)
        self.outside_score_func = Bilinear(self.size)

        if self.compress:
            self.root_mat_out = nn.Parameter(torch.FloatTensor(self.size, self.size))
        else:
            self.root_vector_out_h = nn.Parameter(torch.FloatTensor(self.size))
        self.root_vector_out_c = None

        self.inside_compose_func = ComposeMLP(self.size, self.activation, n_layers=self.n_layers, leaf=True)
        self.outside_compose_func = ComposeMLP(self.size, self.activation, n_layers=self.n_layers)

