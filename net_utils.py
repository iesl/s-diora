import math
import torch
import torch.nn as nn
import numpy as np

from scipy.special import factorial

from outside_index import get_outside_index, get_topk_outside_index, OutsideIndexCheck
from inside_index import get_inside_index, InsideIndexCheck
from inside_index import get_inside_index_unique, get_inside_components
from offset_cache import get_offset_cache

TINY = 1e-8


def nested_del(o, k):
    if isinstance(o[k], dict):
        keys = list(o[k].keys())
        for kk in keys:
            nested_del(o[k], kk)
    del o[k]


class UnitNorm(object):
    def __call__(self, x, p=2, eps=TINY):
        return x / x.norm(p=p, dim=-1, keepdim=True).clamp(min=eps)


class NormalizeFunc(nn.Module):
    def __init__(self, mode='none', size=None):
        super(NormalizeFunc, self).__init__()
        self.mode = mode

    def forward(self, x):
        mode = self.mode
        if mode == 'none':
            return x
        elif mode == 'unit':
            return UnitNorm()(x)
        elif mode == 'layer':
            return nn.functional.layer_norm(x, x.shape[-1:])
        raise Exception('Bad mode = {}'.format(mode))


class BatchInfo(object):
    def __init__(self, **kwargs):
        super(BatchInfo, self).__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)


def build_chart(batch_size, length, size, dtype=None, cuda=False):
    ncells = int(length * (1 + length) / 2)

    device = torch.cuda.current_device() if cuda else None

    chart = {}

    ## Inside.
    chart['inside_h'] = torch.full((batch_size, ncells, size), 0, dtype=dtype, device=device)
    chart['inside_s'] = torch.full((batch_size, ncells, 1), 0, dtype=dtype, device=device)
    chart['inside_error'] = torch.full((batch_size, ncells, 1), 0, dtype=dtype, device=device)

    ## Outside.
    chart['outside_h'] = torch.full((batch_size, ncells, size), 0, dtype=dtype, device=device)
    chart['outside_s'] = torch.full((batch_size, ncells, 1), 0, dtype=dtype, device=device)

    return chart


def get_catalan(n):
    if n > 10:
        return 5000 # HACK: We only use this to check number of trees, and this avoids overflow.
    n = n - 1
    def choose(n, p):
        return factorial(n) / (factorial(p) * factorial(n-p))
    return int(choose(2 * n, n) // (n + 1))


class Index(object):
    def __init__(self, cuda=False, enable_caching=True):
        super(Index, self).__init__()
        self.cuda = cuda
        self.cache = {}
        self.inside_index_cache = {}
        self.inside_index_unique_cache = {}
        self.outside_index_cache = {}
        self.outside_encoded_index_cache = {}
        self.offset_cache = {}
        self.enable_caching = enable_caching

    def cached_lookup(self, func, name, key):
        if name not in self.cache:
            self.cache[name] = {}
        cache = self.cache[name]
        if self.enable_caching:
            if key not in cache:
                cache[key] = func()
            return cache[key]
        else:
            return func()

    def get_catalan(self, n):
        name = 'catalan'
        key = n
        def func():
            return get_catalan(n)
        return self.cached_lookup(func, name, key)

    def get_offset(self, length):
        name = 'offset_cache'
        key = length
        def func():
            return get_offset_cache(length)
        return self.cached_lookup(func, name, key)

    def get_inside_index(self, length, level):
        name = 'inside_index_cache'
        key = (length, level)
        def func():
            return get_inside_index(length, level,
                self.get_offset(length), cuda=self.cuda)
        return self.cached_lookup(func, name, key)

    def get_inside_index_unique(self, length, level):
        name = 'inside_index_unique_cache'
        key = (length, level)
        def func():
            return get_inside_index_unique(length, level,
                self.get_offset(length), cuda=self.cuda)
        return self.cached_lookup(func, name, key)

    def get_outside_index(self, length, level):
        name = 'outside_index_cache'
        key = (length, level)
        def func():
            return get_outside_index(length, level,
                self.get_offset(length), cuda=self.cuda)
        return self.cached_lookup(func, name, key)

    def get_topk_outside_index(self, length, level, K):
        name = 'topk_outside_index_cache'
        key = (length, level, K)
        def func():
            return get_topk_outside_index(length, level, K,
                self.get_offset(length), cuda=self.cuda)
        return self.cached_lookup(func, name, key)


# Inside

def inside_fill_chart(batch_info, chart, index, h=None, s=None, error=None):
    L = batch_info.length - batch_info.level

    offset = index.get_offset(batch_info.length)[batch_info.level]

    if h is not None:
        chart['inside_h'][:, offset:offset+L] = h
    if s is not None:
        chart['inside_s'][:, offset:offset+L] = s
    if error is not None:
        chart['inside_error'][:, offset:offset+L] = error


def get_inside_states(batch_info, chart, index, size):
    lidx, ridx = index.get_inside_index(batch_info.length, batch_info.level)

    ls = chart.index_select(index=lidx, dim=1).view(-1, size)
    rs = chart.index_select(index=ridx, dim=1).view(-1, size)

    return ls, rs


def inside_compose(compose_func, hs):
    return compose_func(hs)


def inside_score(score_func, batch_info, hs):
    return score_func(hs[0], hs[1])


# Outside

def outside_fill_chart(batch_info, chart, index, h, s):
    L = batch_info.length - batch_info.level

    offset = index.get_offset(batch_info.length)[batch_info.level]

    chart['outside_h'][:, offset:offset+L] = h
    chart['outside_s'][:, offset:offset+L] = s


def get_outside_states(batch_info, pchart, schart, index, size):
    pidx, sidx = index.get_outside_index(batch_info.length, batch_info.level)

    ps = pchart.index_select(index=pidx, dim=1).view(-1, size)
    ss = schart.index_select(index=sidx, dim=1).view(-1, size)

    return ps, ss


def outside_compose(compose_func, hs):
    return compose_func(hs, 0)


def outside_score(score_func, batch_info, hs):
    B = batch_info.batch_size
    L = batch_info.length - batch_info.level

    s = score_func(hs[0], hs[1])

    return s

