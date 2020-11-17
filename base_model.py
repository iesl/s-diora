import math
import torch
import torch.nn as nn
import numpy as np

from net_utils import *


class DioraBase(nn.Module):
    r"""DioraBase

    """

    K = 1

    def safe_set_K(self, val):
        pass

    def __init__(self, size=None, outside=True,
                 loss_augmented=False, choose_tree='local', err_scale=1,
                 ninput=2, outside_score='bilinear', component_constraint='submax', **kwargs):
        super(DioraBase, self).__init__()

        self.size = size
        self.err_scale = err_scale
        self.choose_tree = choose_tree
        self.loss_augmented = loss_augmented
        self._outside = outside
        self.outside_score = outside_score
        self.ninput = ninput
        self.inside_normalize_func = NormalizeFunc('unit', size=size)
        self.outside_normalize_func = NormalizeFunc('unit', size=size)
        self.cell_loss = kwargs.get('cell_loss', False)
        self.init = kwargs.get('init', 'normal')
        self.chart_encoding = kwargs.get('chart_encoding', 'none')
        self.compatibility = kwargs.get('compatibility', 'bilinear')
        self.component_constraint = component_constraint

        self.activation = nn.ReLU()

        self.index = None
        self.cache = None
        self.chart = None

        self.init_parameters()
        self.reset_parameters()
        self.reset()

    def init_parameters(self):
        raise NotImplementedError

    def reset_parameters(self):
        if self.init == 'normal':
            params = [p for p in self.parameters() if p.requires_grad]
            for i, param in enumerate(params):
                param.data.normal_()
        elif self.init == 'xavier':
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                if len(param.shape) == 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.zeros_(param)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_cuda(self):
        device = self.device
        return device.index is not None and device.index >= 0

    @property
    def inside_h(self):
        return self.chart['inside_h']

    @property
    def inside_s(self):
        return self.chart['inside_s']

    @property
    def outside_h(self):
        return self.chart['outside_h']

    @property
    def outside_s(self):
        return self.chart['outside_s']

    def cuda(self):
        super(DioraBase, self).cuda()
        if self.index is not None:
            self.index.cuda = True # TODO: Should support to/from cpu/gpu.

    def get(self, chart, level):
        length = self.length
        L = length - level
        offset = self.index.get_offset(length)[level]
        return chart[:, offset:offset+L]

    def leaf_transform(self, x):
        normalize_func = self.inside_normalize_func
        transform_func = self.inside_compose_func.leaf_transform

        input_shape = x.shape[:-1]
        h = transform_func(x)
        h = normalize_func(h.view(*input_shape, self.size))

        return h

    def private_inside_hook(self, level, h, s, p, x_s, l_s, r_s):
        """
        This method is meant to be private, and should not be overriden.
        Instead, override `inside_hook`.
        """
        if level == 0:
            return

        batch_size = self.batch_size
        length = self.length
        B = self.batch_size
        L = length - level
        N = level

        if x_s is not None:
            x_s = x_s.view(*s.shape)

        assert s.shape[0] == B
        assert s.shape[1] == L
        assert s.shape[2] == N
        assert s.shape[3] == 1
        assert len(s.shape) == 4

        if self.component_constraint == 'submax':
            smax = s.max(dim=2, keepdim=True)[0]
            s = s - smax
        elif self.component_constraint == 'local':
            s = x_s

        # TODO: Save this is a single large tensor instead.
        for pos in range(L):
            self.cache['inside_s_components'][level][pos] = s[:, pos, :]

        if x_s is not None:
            for pos in range(L):
                self.cache['inside_xs_components'][level][pos] = x_s[:, pos, :]

        h = h.view(B, L, -1, self.size)
        for pos in range(L):
            self.cache['inside_h_components'][level][pos] = h[:, pos, :]

        # backtrack
        offset_cache = self.index.get_offset(self.length)
        components = get_inside_components(self.length, level, offset_cache)

        component_lookup = {}
        for n_idx, (_, _, x_span, l_span, r_span) in enumerate(components):
            for j, (x_level, x_pos) in enumerate(x_span):
                l_level, l_pos = l_span[j]
                r_level, r_pos = r_span[j]
                component_lookup[(x_pos, n_idx)] = (l_level, l_pos, r_level, r_pos)

        if x_s is not None:
            if self.loss_augmented:
                # Implements Loss Augmented Inference
                index = self.index
                chart = self.chart
                batch_info = BatchInfo(batch_size=batch_size, length=length, size=None, level=level, phase='error')

                # Get children error.
                l_err, r_err = get_inside_states(batch_info, chart['inside_error'], index, 1)

                # Get error.
                x_err = s.clone().fill_(0)
                for i_b in range(batch_size):
                    for pos in range(L):
                        size = level + 1
                        val = 1 if (pos, size) not in self.cache['gold_spans'][i_b] else 0
                        x_err[i_b, pos] = val

                x_err = x_err * self.err_scale

                cumulative_error = l_err.view(*s.shape) + r_err.view(*s.shape) + x_err
                argmax = (x_s + cumulative_error).argmax(dim=2)
                argmax_error = cumulative_error.gather(dim=2, index=argmax.view(B, L, 1, 1)).view(B, L, 1)

                # Update error.
                inside_fill_chart(batch_info, chart, index, error=argmax_error)

            else:
                if self.choose_tree == 'local':
                    argmax = x_s.argmax(dim=2)
                elif self.choose_tree == 'total':
                    argmax = s.view(*x_s.shape).argmax(dim=2)
                elif self.choose_tree == 'history':
                    with torch.no_grad():
                        local_history = x_s.data.clone().fill_(0)
                        for pos in range(L):
                            for n_idx in range(N):
                                l_level, l_pos, r_level, r_pos = component_lookup[(pos, n_idx)]
                                l_idx = self.index.get_offset(length)[l_level] + l_pos
                                r_idx = self.index.get_offset(length)[r_level] + r_pos

                                l_score = self.cache['inside_score_history'][:, l_idx]
                                r_score = self.cache['inside_score_history'][:, r_idx]

                                local_history[:, pos, n_idx] = l_score + r_score
                        local_score = x_s.data.clone() + local_history
                        argval, argmax = local_score.max(dim=2)

                        for pos in range(L):
                            x_idx = self.index.get_offset(length)[level] + pos
                            self.cache['inside_score_history'][:, x_idx] = argval[:, pos]
                else:
                    raise ValueError

            for i_b in range(B):
                for pos in range(L):
                    n_idx = argmax[i_b, pos].item()
                    l_level, l_pos, r_level, r_pos = component_lookup[(pos, n_idx)]

                    self.cache['inside_tree'][(i_b, 0)][(level, pos)] = \
                        self.cache['inside_tree'][(i_b, 0)][(l_level, l_pos)] + \
                        self.cache['inside_tree'][(i_b, 0)][(r_level, r_pos)] + \
                        [(level, pos)]

                    self.cache['inside_tree_edges'][(i_b, 0)][(level, pos)] = \
                        self.cache['inside_tree_edges'][(i_b, 0)][(l_level, l_pos)] + \
                        self.cache['inside_tree_edges'][(i_b, 0)][(r_level, r_pos)] + \
                        [(level, pos, 0, l_level, l_pos, 0, r_level, r_pos, 0)]

    def inside_pass(self):
        compose_func = self.inside_compose_func
        score_func = self.inside_score_func
        index = self.index
        chart = self.chart
        normalize_func = self.inside_normalize_func

        def _score_func(batch_info, hs):
            return inside_score(score_func, batch_info, hs)

        for level in range(1, self.length):

            batch_info = BatchInfo(
                batch_size=self.batch_size,
                length=self.length,
                size=self.size,
                level=level,
                phase='inside',
                )

            self.update_batch_info(batch_info)

            h, s, p, xs, ls, rs = self.inside_func(compose_func, _score_func, batch_info, chart, index,
                normalize_func=normalize_func)

            self.private_inside_hook(level, h, s, p, xs, ls, rs)
            self.inside_hook(level, h, s, p)

    def logsumexp(self, lst):
        tmp = torch.cat(lst, -1)
        d = torch.max(tmp, dim=-1, keepdim=True)[0]
        return torch.log(torch.exp(tmp - d).sum(dim=-1, keepdim=True)) + d

    def inside_func(self, compose_func, score_func, batch_info, chart, index, normalize_func):
        chart_encoding = self.chart_encoding
        device = torch.cuda.current_device() if self.is_cuda else None
        B = batch_info.batch_size
        L = batch_info.length - batch_info.level
        N = batch_info.level
        level = batch_info.level

        lh, rh = get_inside_states(batch_info, chart['inside_h'], index, batch_info.size)
        ls, rs = get_inside_states(batch_info, chart['inside_s'], index, 1)

        h = inside_compose(compose_func, [lh, rh])
        xs = score_func(batch_info, [lh, rh])

        s = xs + ls + rs
        s = s.view(B, L, N, 1)
        p = torch.softmax(s, dim=2)

        hbar = torch.sum(h.view(B, L, N, -1) * p, 2)
        hbar = normalize_func(hbar)
        sbar = torch.sum(s * p, 2)

        inside_fill_chart(batch_info, chart, index, hbar, sbar)

        return h, s, p, xs, ls, rs

    def get_inside_component_index(self, batch_info):
        device = torch.cuda.current_device() if self.is_cuda else None

        B = batch_info.batch_size
        L = batch_info.length - batch_info.level
        N = batch_info.level
        level = batch_info.level

        component_index = torch.tensor([0] * (B * L), dtype=torch.long, device=device)

        for batch_i in range(B):
            for pos in range(L):
                offset = batch_i * L + pos
                component_index[offset] = self.inside_pool_index[batch_i].get((level, pos), 0)

        return component_index

    def soft_inside_score(self):
        pass

    def inside_hook(self, level, h, s, p=None):
        pass

    def outside_hook(self, level, h, s, p=None):
        pass

    def initialize_outside_root(self):
        B = self.batch_size
        D = self.size
        normalize_func = self.outside_normalize_func

        h = self.root_vector_out_h.view(1, 1, D).expand(B, 1, D)

        h = normalize_func(h)

        self.outside_h[:, -1:] = h

    def outside_pass(self):
        self.initialize_outside_root()

        compose_func = self.outside_compose_func
        score_func = self.outside_score_func
        index = self.index
        chart = self.chart
        normalize_func = self.outside_normalize_func

        if self.outside_score == 'bilinear':
            def _score_func(batch_info, hs):
                return outside_score(score_func, batch_info, hs)
        elif self.outside_score == 'soft-inside':
            def _score_func(batch_info, hs, ss):
                B = batch_info.batch_size
                L = batch_info.length - batch_info.level

                pidx, sidx = index.get_outside_index(batch_info.length, batch_info.level)

                I = pidx.shape[0]

                lst = [self.cache['s'][(pidx[i].item(), sidx[i].item())] for i in range(I)]
                s = torch.cat(lst, dim=1).view(B, -1, L, 1)
                lst = [self.cache['p'][(pidx[i].item(), sidx[i].item())] for i in range(I)]
                p = torch.cat(lst, dim=1).view(B, -1, L, 1)

                return s, p
        elif self.outside_score == 'hard-inside':
            def _score_func(batch_info, hs, ss):
                """
                1 if the cells are in the chart, 0 otherwise.
                """
                B = batch_info.batch_size
                L = batch_info.length - batch_info.level
                parse_lookup = self.cache['parse_lookup']

                pidx, sidx = index.get_outside_index(batch_info.length, batch_info.level)
                I = pidx.shape[0]
                keys = [(pidx[i].item(), sidx[i].item()) for i in range(I)]

                mask = []
                for batch_i in range(B):
                    is_constituent = [k in parse_lookup[batch_i] for k in keys]
                    mask.append(torch.LongTensor(is_constituent).view(1, -1))
                mask = torch.cat(mask, 0).float().view(B, -1, L, 1)

                if self.is_cuda:
                    mask = mask.cuda()

                return mask, mask

        for level in range(self.length - 2, -1, -1):
            batch_info = BatchInfo(
                batch_size=self.batch_size,
                length=self.length,
                size=self.size,
                level=level,
                phase='outside',
                )

            self.update_batch_info(batch_info)

            h, s, p, x_s, par_s, sis_s = self.outside_func(compose_func, _score_func, batch_info, chart, index,
                normalize_func=normalize_func)

            self.private_outside_hook(level, h, s, p, x_s, par_s, sis_s)
            self.outside_hook(level, h, s, p)

    def private_outside_hook(self, level, h, s, p, x_s, par_s, sis_s):
        """
        This method is meant to be private, and should not be overriden.
        Instead, override `outside_hook`.
        """
        if level == self.length - 1:
            return

        length = self.length
        B = self.batch_size
        L = length - level

        assert s.shape[0] == B
        # assert s.shape[1] == N
        assert s.shape[2] == L
        assert s.shape[3] == 1
        assert len(s.shape) == 4

        if self.component_constraint == 'submax':
            smax = s.max(dim=1, keepdim=True)[0]
            s = s - smax
        elif self.component_constraint == 'local':
            s = x_s.view(*s.shape)

        # TODO: Save this is a single large tensor instead.
        for pos in range(L):
            self.cache['outside_s_components'][level][pos] = s[:, :, pos]

    def outside_func(self, compose_func, score_func, batch_info, chart, index, normalize_func):
        B = batch_info.batch_size
        L = batch_info.length - batch_info.level

        ph, sh = get_outside_states(
            batch_info, chart['outside_h'], chart['inside_h'], index, batch_info.size)
        ps, ss = get_outside_states(
            batch_info, chart['outside_s'], chart['inside_s'], index, 1)

        h = outside_compose(compose_func, [sh, ph])
        xs = score_func(batch_info, [sh, ph])

        s = xs + ss + ps
        s = s.view(B, -1, L, 1)
        p = torch.softmax(s, dim=1)
        N = s.shape[1]

        hbar = torch.sum(h.view(B, N, L, -1) * p, 1)
        hbar = normalize_func(hbar)
        sbar = torch.sum(s * p, 1)

        outside_fill_chart(batch_info, chart, index, hbar, sbar)

        return h, s, p, xs, ps, ss

    def init_with_batch(self, h, info={}):
        size = self.size
        batch_size, length, _ = h.shape

        self.outside = info.get('outside', self._outside)
        self.inside_pool = info.get('inside_pool', 'sum')
        self.inside_pool_index = info.get('inside_pool_index', {})

        self.batch_size = batch_size
        self.length = length

        self.inside_h[:, :self.length] = h

        # Book-keeping for cky.
        ncells = int(length * (1 + length) / 2)
        self.cache['inside_score_history'] = torch.FloatTensor(batch_size, ncells, 1).to(self.device)
        self.cache['inside_s_components'] = {i: {} for i in range(self.length)}
        self.cache['inside_xs_components'] = {i: {} for i in range(self.length)}
        self.cache['inside_h_components'] = {i: {} for i in range(self.length)}
        self.cache['outside_s_components'] = {i: {} for i in range(self.length)}

        self.cache['inside_tree'] = {}
        for i in range(self.batch_size):
            for i_k in range(self.K):
                tree = {}
                level = 0
                for pos in range(self.length):
                    tree[(level, pos)] = []
                self.cache['inside_tree'][(i, i_k)] = tree

        self.cache['inside_tree_edges'] = {}
        for i_b in range(self.batch_size):
            for i_k in range(self.K):
                tree = {}
                level = 0
                for pos in range(self.length):
                    tree[(level, pos)] = []
                self.cache['inside_tree_edges'][(i_b, i_k)] = tree

        # Book-keeping for cky.
        self.cache['argmax'] = {i: {} for i in range(self.length)}

        # Book-keeping for outside-score.
        if self.outside_score == 'soft-inside':
            self.cache['s'] = {}
            self.cache['p'] = {}

        self.cache['gold_spans'] = None
        if 'constituency_tags' in info:
            def convert(lst):
                return set([(pos, size) for pos, size, label in lst if size > 1])
            self.cache['gold_spans'] = [convert(x) for x in info['constituency_tags']]

    def nested_del(self, o, k):
        if isinstance(o[k], dict):
            keys = list(o[k].keys())
            for kk in keys:
                self.nested_del(o[k], kk)
        del o[k]

    def reset(self):
        self.batch_size = None
        self.length = None
        self.batch_info = None

        if self.chart is not None:
            keys = list(self.chart.keys())
            for k in keys:
                self.nested_del(self.chart, k)
        self.chart = None

        if self.cache is not None:
            keys = list(self.cache.keys())
            for k in keys:
                self.nested_del(self.cache, k)
        self.cache = None

    def initialize(self, x):
        size = self.size
        batch_size, length = x.shape[:2]
        self.chart = build_chart(batch_size, length, size, dtype=torch.float, cuda=self.is_cuda)
        self.cache = {}

    def update_batch_info(self, batch_info):
        self.batch_info = batch_info

    def get_chart_wrapper(self):
        return self

    def post_inside_hook(self):
        pass

    def forward(self, x, info={}):
        if self.index is None:
            self.index = Index(cuda=self.is_cuda)

        self.reset()
        self.initialize(x)

        h = self.leaf_transform(x)

        self.init_with_batch(h, info)
        self.inside_hook(0, h, None, None)
        self.inside_pass()
        self.post_inside_hook()

        if self.outside:
            self.outside_pass()

        return self.chart

