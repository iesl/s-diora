import collections
import itertools

import numpy as np

import torch
import torch.nn as nn

from diora import DioraMLP
from diora import Bilinear
from diora import ComposeMLP
from diora import build_chart

from diora import BatchInfo
from diora import inside_compose, outside_compose
from diora import inside_score, outside_score
from diora import inside_fill_chart, outside_fill_chart
from diora import get_inside_states
from diora import get_inside_components, get_outside_components

from reading import WSJ_CONSTITUENCY_MAPPING, INVERSE_CONSTITUENCY_MAPPING


def default_getattr(o, k, default=None):
    if not hasattr(o, k):
        return default
    return getattr(o, k)


def get_inside_chart_cfg(diora, new_chart=False, level=None, K=None, device=None, **kwargs):
    batch_size, length, size = diora.batch_size, diora.length, diora.size
    # ChartUtils
    parameters = {}

    inputs = {}
    inputs['device'] = device
    inputs['batch_size'] = diora.batch_size
    inputs['length'] = diora.length
    inputs['size'] = diora.size
    inputs['index'] = diora.index
    inputs['score_func'] = diora.inside_score_func
    inputs['compose_func'] = diora.inside_compose_func
    inputs['normalize_func'] = diora.inside_normalize_func
    inputs['label_func'] = default_getattr(diora, 'label_func', None)
    inputs['training'] = diora.training
    inputs['lse'] = default_getattr(diora, 'lse', False)
    inputs['gold_spans'] = None
    inputs['loss_augmented'] = False

    if new_chart:
        charts = []
        for i in range(K):
            if i == 0:
                charts.append(
                    build_inside_chart(batch_size, length, size, params=dict(init_h=kwargs.get('init_h', None)), dtype=torch.float, device=device))
            else:
                charts.append(
                    build_inside_chart(batch_size, length, size, params=dict(init_h=None), dtype=torch.float, device=device))
                # Never should be selected.
                charts[i]['inside_s'][:] = -1e8
            charts[i]['outside_s'] = None
            charts[i]['outside_h'] = None
        inputs['charts'] = charts
    else:
        inputs['charts'] = diora.charts

    cfg = {}
    cfg['topk'] = K
    cfg['mode'] = 'inside'
    cfg['dense_compose'] = diora.dense_compose
    cfg['labeled'] = diora.labeled

    if level is not None:
        cfg['level'] = level

    return parameters, inputs, cfg


def get_outside_chart_cfg(diora, inside_charts, new_chart=False, level=None, K=None, device=None, **kwargs):
    batch_size, length, size = diora.batch_size, diora.length, diora.size
     # ChartUtils
    params = {}

    inputs = {}
    inputs['device'] = device
    inputs['batch_size'] = batch_size
    inputs['length'] = length
    inputs['size'] = size
    inputs['inside_charts'] = inside_charts

    inputs['root_vector_out_h'] = diora.root_vector_out_h
    inputs['outside_normalize_func'] = diora.outside_normalize_func
    inputs['index'] = diora.index
    inputs['score_func'] = diora.outside_score_func
    inputs['compose_func'] = diora.outside_compose_func
    inputs['normalize_func'] = diora.outside_normalize_func
    inputs['sibling_dropout_dist'] = diora.sibling_dropout_dist
    inputs['coordinate_embeddings_size'] = diora.coordinate_embeddings_size
    inputs['project_coordinate_embedding'] = diora.project_coordinate_embedding
    inputs['lse'] = diora.lse
    inputs['training'] = diora.training

    build_chart_params = {}
    build_chart_params['root_vector_out_h'] = inputs['root_vector_out_h']
    build_chart_params['outside_normalize_func'] = inputs['outside_normalize_func']

    if new_chart:
        outside_charts = []
        for i in range(K):
            outside_charts.append(build_outside_chart(batch_size, length, size, params=build_chart_params, dtype=torch.float, device=device))
            outside_charts[i]['inside_s'] = None
            outside_charts[i]['inside_h'] = None
            if i > 0:
                # Never should be selected.
                outside_charts[i]['outside_s'][:] = -1e8
        inputs['outside_charts'] = outside_charts
    else:
        inputs['outside_charts'] = inside_charts

    cfg = {}
    cfg['topk'] = K
    cfg['mode'] = 'outside'

    if level is not None:
        cfg['level'] = level

    return params, inputs, cfg


def build_inside_correction(inputs, tree_edges, K):
    batch_size, length, size = inputs['batch_size'], inputs['length'], inputs['size']
    index = inputs['index']

    def flatten_edges():
        out = {}
        for i_b, i_k in tree_edges.keys():
            edges = tree_edges[(i_b, i_k)]
            for e in edges:
                x_level, x_pos, x_k, l_level, l_pos, l_k, r_level, r_pos, r_k = e
                # The first outside should always come from the primary chart.
                if l_level == 0:
                    assert l_k == 0, e

                if r_level == 0:
                    assert r_k == 0, e

                assert x_level < length
                assert x_level + 1 == (l_level + 1) + (r_level + 1)

                key = (i_b, x_level, x_pos, i_k)
                assert key not in out
                out[key] = l_level, l_pos, l_k, r_level, r_pos, r_k
        return out

    correction = {}

    target_to_dependents = flatten_edges()

    lookup = {}
    for i_k in range(K):
        for level in range(length):
            offset_cache = index.get_offset(length)
            components = get_inside_components(length, level, offset_cache)
            for idx, (_, _, x_span, l_span, r_span) in enumerate(components):
                for j, (x_level, x_pos) in enumerate(x_span):
                    l_level, l_pos = l_span[j]
                    r_level, r_pos = r_span[j]

                    for i_b in range(batch_size):
                        key_x, key_y = (i_b, level, x_pos, i_k), (l_level, l_pos, r_level, r_pos)
                        key = (key_x, key_y)
                        val = idx
                        assert key not in lookup
                        lookup[key] = val

    correction = {}
    for key_x in target_to_dependents.keys():
        l_level, l_pos, l_k, r_level, r_pos, r_k = target_to_dependents[key_x]
        key_y = l_level, l_pos, r_level, r_pos
        idx = lookup[(key_x, key_y)]
        assert key_x not in correction
        correction[key_x] = (idx, l_level, l_pos, l_k, r_level, r_pos, r_k)

    return correction


def build_inside_correction_spans(inputs, spans, K):
    batch_size = inputs['batch_size']
    correction = {i_b: spans[i_b] for i_b in range(batch_size)}
    return correction


def build_outside_correction(inputs, tree_edges, K):
    batch_size, length = inputs['batch_size'], inputs['length']
    index = inputs['index']

    def flatten_edges_for_outside():
        out = {}
        for i_b, i_k in tree_edges.keys():
            edges = tree_edges[(i_b, i_k)]
            for x_level, x_pos, x_k, l_level, l_pos, l_k, r_level, r_pos, r_k in edges:
                p_k = i_k
                p_level = x_level
                p_pos = x_pos

                # The first outside should always come from the primary chart.
                if p_level == length - 1:
                    p_k = 0

                # Left is target.
                key = (i_b, l_level, l_pos, i_k)
                assert key not in out
                out[key] = p_level, p_pos, p_k, r_level, r_pos, r_k

                # Right is target.
                key = (i_b, r_level, r_pos, i_k)
                assert key not in out
                out[key] = p_level, p_pos, p_k, l_level, l_pos, l_k
        return out

    target_to_parent_and_sibling = flatten_edges_for_outside()

    lookup = {}
    for i_k in range(K):
        for level in range(length):
            L = length - level
            offset_cache = index.get_offset(length)
            components = get_outside_components(length, level, offset_cache)
            for i, (p_span, s_span) in enumerate(components):
                p_level, p_pos = p_span
                s_level, s_pos = s_span
                idx = i // L
                x_pos = i % L
                for i_b in range(batch_size):
                    key_x, key_y = (i_b, level, x_pos, i_k), (p_level, p_pos, s_level, s_pos)
                    key = (key_x, key_y)
                    val = idx
                    assert key not in lookup
                    lookup[key] = val

    correction = {}
    for key_x in target_to_parent_and_sibling.keys():
        p_level, p_pos, p_k, s_level, s_pos, s_k = target_to_parent_and_sibling[key_x]
        key_y = p_level, p_pos, s_level, s_pos
        idx = lookup[(key_x, key_y)]
        assert key_x not in correction
        correction[key_x] = (idx, p_level, p_pos, p_k, s_level, s_pos, s_k)

    return correction


def build_inside_chart(batch_size, length, size, params=None, dtype=None, device=None):
    ncells = int(length * (1 + length) / 2)

    chart = {}
    chart['inside_h'] = torch.full((batch_size, ncells, size), 0, dtype=dtype, device=device)
    chart['inside_s'] = torch.full((batch_size, ncells, 1), 0, dtype=dtype, device=device)
    if params is not None and params.get('init_h', None) is not None:
        chart['inside_h'][:, :length] = params['init_h']
    chart['inside_error'] = torch.full((batch_size, ncells, 1), 0, dtype=dtype, device=device)

    return chart


def build_outside_chart(batch_size, length, size, params=None, dtype=None, device=None):
    ncells = int(length * (1 + length) / 2)

    chart = {}

    ## Outside.
    chart['outside_h'] = torch.full((batch_size, ncells, size), 0, dtype=dtype, device=device)
    chart['outside_s'] = torch.full((batch_size, ncells, 1), 0, dtype=dtype, device=device)

    h = params['root_vector_out_h'].view(1, 1, size).expand(batch_size, 1, size)
    h = params['outside_normalize_func'](h)
    chart['outside_h'][:, -1:] = h

    return chart


def convert_edges_to_spans(edges):
    spans = []
    for e in edges:
        x_level, x_pos, x_k, l_level, l_pos, l_k, r_level, r_pos, r_k = e
        x_size = x_level + 1
        spans.append((x_pos, x_size))
    return spans


class ChartUtil(nn.Module):
    debug = False

    def __init__(self, parameters):
        super(ChartUtil, self).__init__()
        for k, v in parameters.items():
            self.k = v

    def run(self, inputs, cfg):
        if cfg['mode'] == 'inside':
            return self.f_inside(inputs, cfg)
        elif cfg['mode'] == 'outside':
            return self.f_outside(inputs, cfg)

    @staticmethod
    def verify_inside_trees(charts, index, edges, batch_size, length, K):
        if not ChartUtil.debug:
            return

        max_possible_trees = index.get_catalan(length)

        offset_cache = index.get_offset(length)
        for i_b in range(batch_size):
            for i_k in range(min(K, max_possible_trees)):
                val = charts[i_k]['inside_s'][i_b, -1].item()
                assert val > -1e5, (val, i_b, i_k)

                for e in edges[(i_b, i_k)]:
                    x_level, x_pos, x_k, l_level, l_pos, l_k, r_level, r_pos, r_k = e
                    x_idx = offset_cache[x_level] + x_pos
                    l_idx = offset_cache[l_level] + l_pos
                    r_idx = offset_cache[r_level] + r_pos

                    if l_level == 0:
                        assert l_k == 0, e

                    if r_level == 0:
                        assert r_k == 0, e

                    val = charts[x_k]['inside_s'][i_b, x_idx].item()
                    assert val > -1e5, (val, i_b, i_k, x_idx, x_k, e)
                    val = charts[l_k]['inside_s'][i_b, l_idx].item()
                    assert val > -1e5, (val, i_b, i_k, l_idx, l_k, e)
                    val = charts[r_k]['inside_s'][i_b, r_idx].item()
                    assert val > -1e5, (val, i_b, i_k, r_idx, r_k, e)

    @staticmethod
    def verify_correction_spans(batch_size, correction_spans):
        if not ChartUtil.debug:
            return

        def is_crossing(pos0, size0, pos1, size1):
            assert pos0 < pos1, "This check has an order constraint."
            new_pos0 = pos0 + size0
            if new_pos0 > pos1 and new_pos0 < pos1 + size1:
                return True
            return False
        for i_b in range(batch_size):
            for pos0, size0 in correction_spans[i_b]:
                for pos1, size1 in correction_spans[i_b]:
                    if pos0 < pos1:
                        assert is_crossing(pos0, size0, pos1, size1) == False

    @staticmethod
    def verify_span_constraints_satisfied(batch_size, length, chart_output, spans):
        root_level, root_pos, i_k = length - 1, 0, 0
        output_edges = [chart_output['inside_tree_edges'][(i_b, i_k)][(root_level, root_pos)] for i_b in range(batch_size)]
        output_spans = [convert_edges_to_spans(x) for x in output_edges]
        # Verify new tree has constraint spans.
        for i_b in range(batch_size):
            check = set(output_spans[i_b])
            for span in spans[i_b]:
                if span not in check:
                    print(i_b, span)
                    print('constraint:', sorted(spans[i_b]))
                    print('    output:', sorted(output_spans[i_b]))
                assert span in check

    @staticmethod
    def verify_outside_correction(charts, index, correction, batch_size, length, K):
        if not ChartUtil.debug:
            return

        # TODO: Would be easier if correction was dictionary of {(i_b, i_k): edge-constraints}.
        max_possible_trees = index.get_catalan(length)

        offset_cache = index.get_offset(length)

        seen = {i_b: set() for i_b in range(batch_size)}

        for k, v in correction.items():
            i_b, x_level, x_pos, i_k = k

            # Only check for up to K constraints.
            if i_k + 1 > K:
                continue

            n_idx, p_level, p_pos, p_k, s_level, s_pos, s_k = v
            x_idx = offset_cache[x_level] + x_pos
            p_idx = offset_cache[p_level] + p_pos
            s_idx = offset_cache[s_level] + s_pos

            if max_possible_trees > i_k:
                # The first inside should always come from the primary chart.
                if s_level == 0:
                    assert s_k == 0, (max_possible_trees, k, v)

                # The first outside should always come from the primary chart.
                if p_level == length - 1:
                    assert p_k == 0, (max_possible_trees, k, v)

                val = charts[s_k]['inside_s'][i_b, s_idx].item()
                assert val > -1e5, (val, i_k, s_idx, s_k)
                # Doesn't make sense to check these.
                # val = charts[i_k]['outside_s'][i_b, x_idx].item()
                # assert val > -1e5, (val, i_k, x_idx, '_')
                # val = charts[p_k]['outside_s'][i_b, p_idx].item()
                # assert val > -1e5, (val, i_k, p_idx, p_k)

            seen[i_b].add((x_level, x_pos))

        for k, v in correction.items():
            i_b, x_level, x_pos, i_k = k

            # Only check for up to K constraints.
            if i_k + 1 > K:
                continue

            n_idx, p_level, p_pos, p_k, s_level, s_pos, s_k = v
            local_seen = seen[i_b]

            if p_level == length - 1:
                assert p_k == 0
                continue

            assert (p_level, p_pos) in local_seen

    def run_outside_backtrack(self, inputs, cfg):
        device = inputs['device']
        B = inputs['batch_size']
        batch_size = inputs['batch_size']
        length = inputs['length']
        size = inputs['size']
        K = cfg['topk']

        index = inputs['index']
        topk_n_idx = inputs['outside_topk_n_idx']
        topk_pk = inputs['outside_topk_pk']
        topk_sk = inputs['outside_topk_sk']

        saved_trees = {}

        _, _, pairs = ChartUtil.get_tensor_product_mask(K)
        component_lookup = {}
        offset_cache = index.get_offset(length)
        for x_level in range(length):
            L = length - x_level
            N = length - x_level - 1
            components = get_outside_components(length, x_level, offset_cache)
            for i, (p_span, s_span) in enumerate(components):
                p_level, p_pos = p_span
                s_level, s_pos = s_span
                idx = i // L
                x_pos = i % L
                component_lookup[(x_level, x_pos, idx)] = (p_level, p_pos, s_level, s_pos)

        def helper(i_b, level, pos, n_idx, pk, sk):
            key = (level, pos, n_idx)

            p_level, p_pos, s_level, s_pos = component_lookup[key]

            if p_level < length - 1:
                next_n_idx = inputs['outside_topk_n_idx'][p_level][p_pos][i_b, pk].item()
                next_pk = inputs['outside_topk_pk'][p_level][p_pos][i_b, pk].item()
                next_sk = inputs['outside_topk_sk'][p_level][p_pos][i_b, pk].item()

                parent_tree = helper(i_b, p_level, p_pos, next_n_idx, next_pk, next_sk)
            else:
                parent_tree = []

            sibling_tree = inputs['inside_tree'][(i_b, sk)][(s_level, s_pos)]
            tree = [(p_level, p_pos)] + parent_tree + sibling_tree
            return tree

        for i_b in range(batch_size):
            for i_k in range(K):
                for i_pos in range(length):
                    leaf_level = 0
                    leaf_pos = i_pos
                    leaf_n_idx = topk_n_idx[leaf_level][leaf_pos][i_b, i_k].item()
                    leaf_pk = topk_pk[leaf_level][leaf_pos][i_b, i_k].item()
                    leaf_sk = topk_sk[leaf_level][leaf_pos][i_b, i_k].item()
                    tree = helper(i_b, leaf_level, leaf_pos, leaf_n_idx, leaf_pk, leaf_sk)
                    saved_trees[(i_b, i_pos, i_k)] = tree

        return saved_trees

    def f_inside(self, inputs, cfg):
        outputs = {}
        outputs['by_level'] = {}
        outputs['inside_tree_edges'] = {}

        batch_size = inputs['batch_size']
        length = inputs['length']
        K = cfg['topk']

        for i_b in range(batch_size):
            for i_k in range(K):
                tree = {}
                level = 0
                for pos in range(inputs['length']):
                    tree[(level, pos)] = []
                outputs['inside_tree_edges'][(i_b, i_k)] = tree

        if cfg.get('level', None) is None:
            for level in range(1, inputs['length']):
                local_cfg = cfg.copy()
                local_cfg['level'] = level
                batch_info = BatchInfo(batch_size=inputs['batch_size'], length=inputs['length'],
                    size=inputs['size'], level=level, phase='inside')
                outputs['by_level'][level] = \
                    self.f_hard_inside_helper(inputs, outputs, local_cfg, batch_info)

                # backtrack
                B = batch_size
                L = length - level
                N = level

                offset_cache = inputs['index'].get_offset(length)
                components = get_inside_components(length, level, offset_cache)

                component_lookup = {}
                for idx, (_, _, x_span, l_span, r_span) in enumerate(components):
                    for j, (x_level, x_pos) in enumerate(x_span):
                        l_level, l_pos = l_span[j]
                        r_level, r_pos = r_span[j]
                        component_lookup[(x_pos, idx)] = (l_level, l_pos, r_level, r_pos)

                topk_n_idx = outputs['by_level'][level]['topk_n_idx']
                topk_lk = outputs['by_level'][level]['topk_lk']
                topk_rk = outputs['by_level'][level]['topk_rk']

                for i_b in range(batch_size):
                    for pos in range(L):
                        for i_k in range(K):
                            n_idx = topk_n_idx[i_b, pos, i_k].item()
                            l_k = topk_lk[i_b, pos, i_k].item()
                            r_k = topk_rk[i_b, pos, i_k].item()
                            l_level, l_pos, r_level, r_pos = component_lookup[(pos, n_idx)]

                            outputs['inside_tree_edges'][(i_b, i_k)][(level, pos)] = \
                                outputs['inside_tree_edges'][(i_b, l_k)][(l_level, l_pos)] + \
                                outputs['inside_tree_edges'][(i_b, r_k)][(r_level, r_pos)] + \
                                    [(level, pos, i_k, l_level, l_pos, l_k, r_level, r_pos, r_k)]

                if cfg.get('level_hook', None) is not None:
                    cfg['level_hook'](batch_info=batch_info, level_output=outputs['by_level'][level])
        else:
            level = cfg['level']
            batch_info = BatchInfo(batch_size=inputs['batch_size'], length=inputs['length'],
                size=inputs['size'], level=level, phase='inside')
            outputs['by_level'][level] = \
                self.f_hard_inside_helper(inputs, None, cfg, batch_info)

        return outputs

    def f_outside(self, inputs, cfg):
        outputs = {}
        outputs['by_level'] = {}

        if cfg.get('level', None) is None:
            for level in range(0, inputs['length'] - 1)[::-1]:
                local_cfg = cfg.copy()
                local_cfg['level'] = level
                batch_info = BatchInfo(batch_size=inputs['batch_size'], length=inputs['length'],
                    size=inputs['size'], level=level, phase='outside')
                outputs['by_level'][level] = \
                    self.f_hard_outside_helper(inputs, local_cfg, batch_info)

                if cfg.get('level_hook', None) is not None:
                    cfg['level_hook'](batch_info=batch_info, level_output=outputs['by_level'][level])
        else:
            level = cfg['level']
            batch_info = BatchInfo(batch_size=inputs['batch_size'], length=inputs['length'],
                size=inputs['size'], level=level, phase='outside')
            outputs['by_level'][level] = \
                self.f_hard_outside_helper(inputs, cfg, batch_info)

        return outputs

    @staticmethod
    def get_tensor_product_mask(K):
        l_prod = []
        for i0 in range(K):
            lst = []
            for i1 in range(K):
                for i2 in range(K):
                    if i0 == i1:
                        lst.append(1)
                    else:
                        lst.append(0)
            l_prod.append(tuple(lst))
        l_prod = tuple(l_prod)

        r_prod = []
        for i0 in range(K):
            lst = []
            for i1 in range(K):
                for i2 in range(K):
                    if i0 == i2:
                        lst.append(1)
                    else:
                        lst.append(0)
            r_prod.append(tuple(lst))
        r_prod = tuple(r_prod)

        pairs = np.concatenate([
            np.array(l_prod).argmax(axis=0).reshape(1, -1),
            np.array(r_prod).argmax(axis=0).reshape(1, -1)], axis=0).T.tolist()

        return l_prod, r_prod, pairs

    @staticmethod
    def convert_to_idx(n_idx, lk, rk, N, K):
        # topk_n_idx = topk_idx // (K**2)
        # topk_l_k = topk_idx % (K**2) // K
        # topk_r_k = topk_idx % (K**2) % K
        return n_idx * K * K + lk * K + rk

    @staticmethod
    def outside_convert_to_idx(n_idx, pk, sk, N, K):
        return n_idx * K * K + pk * K + sk

    def f_hard_inside_helper(self, inputs, outputs, cfg, batch_info):
        # Note: Outputs should be read-only.
        device = inputs['device']
        B = inputs['batch_size']
        L = inputs['length'] - cfg['level']
        N = cfg['level']
        size = inputs['size']
        level = cfg['level']
        batch_size = inputs['batch_size']
        length = inputs['length']
        training = inputs['training']

        index = inputs['index']
        charts = inputs['charts']
        CH = {}
        compose_func = inputs['compose_func']
        lse = inputs['lse']
        normalize_func = inputs['normalize_func']
        label_func = inputs['label_func']
        loss_augmented = inputs['loss_augmented']

        K = cfg['topk']
        force = cfg.get('force', False)
        correction = cfg.get('correction', None)
        correction_spans = cfg.get('correction_spans', None)
        dense_compose = cfg.get('dense_compose', False)
        labeled = cfg.get('labeled', False)

        def score_func(batch_info, hs):
            return inside_score(inputs['score_func'], batch_info, hs)

        offset_cache = index.get_offset(batch_info.length)
        components = get_inside_components(batch_info.length, batch_info.level, offset_cache)
        l_prod, r_prod, pairs = self.get_tensor_product_mask(K)

        component_lookup = {}
        for n_idx, (_, _, x_span, l_span, r_span) in enumerate(components):
            for j, (x_level, x_pos) in enumerate(x_span):
                l_level, l_pos = l_span[j]
                r_level, r_pos = r_span[j]
                component_lookup[(x_pos, n_idx)] = (l_level, l_pos, r_level, r_pos)

        # DIORA.

        assert len(charts) == K

        for i, chart in enumerate(charts):
            lh, rh = get_inside_states(batch_info, chart['inside_h'], index, batch_info.size)
            CH.setdefault('lh', []).append(lh)
            CH.setdefault('rh', []).append(rh)

            ls, rs = get_inside_states(batch_info, chart['inside_s'], index, 1)
            CH.setdefault('ls', []).append(ls)
            CH.setdefault('rs', []).append(rs)

            if loss_augmented:
                l_err, r_err = get_inside_states(batch_info, chart['inside_error'], index, 1)
                CH.setdefault('l_err', []).append(l_err)
                CH.setdefault('r_err', []).append(r_err)

        ## All Combos

        mat = inputs['score_func'].mat

        # a : (B, L, N, K, D)
        # b : (B, L, N, D, K)
        # out : (B, L, N, K, K)
        lh = torch.cat([CH['lh'][i].view(B, L, N, 1, size) for i in range(K)], 3)
        rh = torch.cat([CH['rh'][i].view(B, L, N, size, 1) for i in range(K)], 4)
        s_raw = torch.matmul(torch.matmul(lh, mat), rh)
        if lse:
            s_raw = torch.nn.functional.logsigmoid(s_raw)

        select_ls = torch.tensor(l_prod, dtype=torch.float, device=device)
        select_rs = torch.tensor(r_prod, dtype=torch.float, device=device)

        # a : (B, L, N, K)
        # b : (B, L, N, K)
        ls = torch.cat([CH['ls'][i].view(B, L, N, 1) for i in range(K)], 3)
        rs = torch.cat([CH['rs'][i].view(B, L, N, 1) for i in range(K)], 3)
        combo_ls = torch.matmul(ls, select_ls).view(B, L, N * K * K, 1)
        combo_rs = torch.matmul(rs, select_rs).view(B, L, N * K * K, 1)
        combo_s = s_raw.view(B, L, N * K * K, 1)

        def logsumexp(a, b, c, dim=1):
            v = torch.cat([a.unsqueeze(dim), b.unsqueeze(dim), c.unsqueeze(dim)], dim)
            d = torch.max(v, dim=dim)[0]
            out = torch.log(torch.exp(v - d.unsqueeze(dim)).sum(dim)) + d
            out[(b < -1e5) | (c < -1e5)] = -1e8
            return out

        if lse:
            s = logsumexp(combo_s, combo_ls, combo_rs, dim=1)
        else:
            s = combo_s + combo_ls + combo_rs
        local_s = combo_s

        # We should not include any split that includes an in-complete beam.
        def penalize_incomplete_splits():
            force_index = collections.defaultdict(list)
            for i_b in range(batch_size):
                for pos in range(L):
                    for i, (l_k, r_k) in enumerate(pairs):
                        idx = i // (K**2)
                        l_level, l_pos, r_level, r_pos = component_lookup[(pos, idx)]
                        max_possible_trees = index.get_catalan(l_level + 1)
                        if max_possible_trees < l_k + 1:
                            force_index['i_b'].append(i_b)
                            force_index['pos'].append(pos)
                            force_index['i'].append(i)
                        max_possible_trees = index.get_catalan(r_level + 1)
                        if max_possible_trees < r_k + 1:
                            force_index['i_b'].append(i_b)
                            force_index['pos'].append(pos)
                            force_index['i'].append(i)
            s[force_index['i_b'], force_index['pos'], force_index['i']] = -1e8

        if not force:
            # Don't do this when the trees are specified since it's possible for duplicate
            # sub-trees to exist.
            penalize_incomplete_splits()

        # Aggregate.
        if loss_augmented:
            l_err = torch.cat([CH['l_err'][i].view(B, L, N, 1) for i in range(K)], 3)
            r_err = torch.cat([CH['r_err'][i].view(B, L, N, 1) for i in range(K)], 3)
            x_err = s.clone().fill_(0)
            for i_b in range(batch_size):
                for pos in range(L):
                    x_size = level + 1
                    x_err[i_b, pos] = 0 if (pos, x_size) in inputs['gold_spans'][i_b] else 1

            cumulative_error = x_err + l_err + r_err
            choose_s = s + cumulative_error
            _, topk_idx = choose_s.topk(dim=2, k=K)
            topk_s = s.gather(dim=2, index=topk_idx)
            topk_error = cumulative_error.gather(dim=2, index=topk_idx)

        else:
            topk_s, topk_idx = s.topk(dim=2, k=K)

        # Force topk according to given constraints.
        force_info = None
        if force and correction_spans is not None: # force spans
            assert s.shape == (B, L, N * K * K, 1)

            force_info = {}
            force_info['found_valid'] = {}
            force_info['idx_invalid'] = collections.defaultdict(set)
            force_info['found_contains'] = {}
            force_info['idx_notcontain'] = collections.defaultdict(set)
            force_info['found_crossing'] = {}
            force_info['idx_crossing'] = collections.defaultdict(set)

            def is_crossing(i_b, y_level, y_pos):
                y_size = y_level + 1
                if y_size == 1:
                    return False

                for x_pos, x_size in correction_spans[i_b]:

                    def helper(pos0, size0, pos1, size1):
                        assert pos0 < pos1, "This check has an order constraint."
                        new_pos0 = pos0 + size0
                        if new_pos0 > pos1 and new_pos0 < pos1 + size1:
                            return True
                        return False

                    if y_pos < x_pos and helper(y_pos, y_size, x_pos, x_size):
                        return True
                    if x_pos < y_pos and helper(x_pos, x_size, y_pos, y_size):
                        return True
                return False

            def does_contain(i_b, y_level, y_pos, y_k):
                return (i_b, y_level, y_pos, y_k) in outputs['by_level'][y_level]['contains']

            for n_idx, (_, _, x_span, l_span, r_span) in enumerate(components):
                for j, (x_level, x_pos) in enumerate(x_span):
                    l_level, l_pos = l_span[j]
                    l_size = l_level + 1
                    r_level, r_pos = r_span[j]
                    r_size = r_level + 1

                    # Check for inclusion (must be a constraint and sibling does not violate constraint).
                    for i_b in range(batch_size):
                        found = False
                        if (l_pos, l_size) in correction_spans[i_b]:
                            if not is_crossing(i_b, r_level, r_pos):
                                found = True
                        if (r_pos, r_size) in correction_spans[i_b]:
                            if not is_crossing(i_b, l_level, l_pos):
                                found = True
                        if found:
                            force_info['found_valid'][(i_b, x_pos)] = True
                        else:
                            for l_k in range(K):
                                for r_k in range(K):
                                    idx = ChartUtil.convert_to_idx(n_idx, l_k, r_k, N, K)
                                    force_info['idx_invalid'][(i_b, x_pos)].add(idx)

                    if level > 1:
                        for i_b in range(batch_size):
                            for l_k in range(K):
                                for r_k in range(K):
                                    found = False
                                    if l_level > 0 and does_contain(i_b, l_level, l_pos, l_k):
                                        if not is_crossing(i_b, r_level, r_pos):
                                            found = True
                                    if r_level > 0 and does_contain(i_b, r_level, r_pos, r_k):
                                        if not is_crossing(i_b, l_level, l_pos):
                                            found = True
                                    if found:
                                        force_info['found_contains'][(i_b, x_pos)] = True
                                    else:
                                        idx = ChartUtil.convert_to_idx(n_idx, l_k, r_k, N, K)
                                        force_info['idx_notcontain'][(i_b, x_pos)].add(idx)

                    # Check for crossing (if any sibling violates constraint).
                    for i_b in range(batch_size):
                        if is_crossing(i_b, l_level, l_pos) or is_crossing(i_b, r_level, r_pos):
                            force_info['found_crossing'][(i_b, x_pos)] = True
                            for l_k in range(K):
                                for r_k in range(K):
                                    idx = ChartUtil.convert_to_idx(n_idx, l_k, r_k, N, K)
                                    force_info['idx_crossing'][(i_b, x_pos)].add(idx)

            # Note: Only one of valid or (crossing + contains) is applied.
            force_index = collections.defaultdict(list)

            for i_b in range(batch_size):
                for x_pos in range(L):
                    # If this span is crossing, then no need to apply constraints.
                    if is_crossing(i_b, level, x_pos):
                        continue

                    if (i_b, x_pos) in force_info['found_valid']:
                        local_invalid = list(force_info['idx_invalid'][(i_b, x_pos)])
                        if len(local_invalid) == 0:
                            continue
                        assert len(force_info['idx_invalid'][(i_b, x_pos)]) < N * K * K
                        for i_invalid in local_invalid:
                            force_index['i_b'].append(i_b)
                            force_index['x_pos'].append(x_pos)
                            force_index['i_invalid'].append(i_invalid)
                        continue

                    if (i_b, x_pos) in force_info['found_contains']:
                        local_invalid = list(force_info['idx_notcontain'][(i_b, x_pos)])
                        if len(local_invalid) == 0:
                            continue
                        assert len(force_info['idx_notcontain'][(i_b, x_pos)]) < N * K * K
                        for i_invalid in local_invalid:
                            force_index['i_b'].append(i_b)
                            force_index['x_pos'].append(x_pos)
                            force_index['i_invalid'].append(i_invalid)
                        continue

                    if (i_b, x_pos) in force_info['found_crossing']:
                        local_invalid = list(force_info['idx_crossing'][(i_b, x_pos)])
                        if len(local_invalid) == 0:
                            continue
                        for i_invalid in local_invalid:
                            force_index['i_b'].append(i_b)
                            force_index['x_pos'].append(x_pos)
                            force_index['i_invalid'].append(i_invalid)

            local_s = s[force_index['i_b'], force_index['x_pos'], force_index['i_invalid']]
            new_s = local_s.clone().fill_(-1e6)
            s[force_index['i_b'], force_index['x_pos'], force_index['i_invalid']] = torch.where(local_s < -1e6, local_s, new_s)

            topk_s, topk_idx = s.topk(dim=2, k=K)


        if force and correction is not None: # force edges
            max_possible_trees = index.get_catalan(length)

            assert length - 1 in correction, "Make sure correction is keyed with levels."

            force_index = collections.defaultdict(list)

            if level in correction:
                for k, v in correction[level]:
                    i_b, level, pos, i_k = k
                    n_idx, l_level, l_pos, l_k, r_level, r_pos, r_k = v

                    new_idx = ChartUtil.convert_to_idx(n_idx, l_k, r_k, N, K)

                    force_index['i_b'].append(i_b)
                    force_index['pos'].append(pos)
                    force_index['i_k'].append(i_k)
                    force_index['new_idx'].append(new_idx)

                    if max_possible_trees > i_k:
                        val = combo_ls[i_b, pos, new_idx].item()
                        assert val > -1e5, (val, l_level, l_pos, l_k)
                        val = combo_rs[i_b, pos, new_idx].item()
                        assert val > -1e5, (val, r_level, r_pos, r_k)
                        val = s[i_b, pos, new_idx].item()
                        assert val > -1e5, (val, level, pos, i_k)

            local_new_idx = torch.tensor(force_index['new_idx'], dtype=torch.long, device=device).unsqueeze(1)
            topk_idx[force_index['i_b'], force_index['pos'], force_index['i_k']] = local_new_idx

            topk_s = s.gather(index=topk_idx, dim=2)

        # Selective compose.
        topk_n = topk_idx // (K**2)
        topk_l_k = topk_idx % (K**2) // K
        topk_l = topk_n * K + topk_l_k
        topk_r_k = topk_idx % (K**2) % K
        topk_r = topk_n * K + topk_r_k

        def _dense_compose():
            lh = torch.cat([CH['lh'][i].view(B, L, N, 1, size) for i in range(K)], 3)
            rh = torch.cat([CH['rh'][i].view(B, L, N, 1, size) for i in range(K)], 3)

            combo_lh = torch.einsum('blnkd,kz->blnzd', lh, select_ls).contiguous().view(-1, size)
            combo_rh = torch.einsum('blnkd,kz->blnzd', rh, select_rs).contiguous().view(-1, size)
            all_h = inside_compose(compose_func, [combo_lh, combo_rh]).view(B, L, N * K * K, size)

            topk_h = all_h.gather(index=topk_idx.expand(B, L, K, size), dim=2)
            topk_h = normalize_func(topk_h)

            all_idx = torch.tensor(range(N * K * K), dtype=torch.long, device=device)\
                .view(1, 1, N * K * K, 1).expand(B, L, N * K * K, 1)

            # Apply topk constraints.
            # TODO: This should be done w/o for loop.
            topk_n_idx = all_idx // (K**2)
            topk_lk = all_idx % (K**2) // K
            topk_rk = all_idx % (K**2) % K

            is_valid = []
            for i_b in range(B):
                for pos in range(L):
                    for n_idx, lk, rk in zip(
                        topk_n_idx[i_b, pos].view(-1).tolist(),
                        topk_lk[i_b, pos].view(-1).tolist(),
                        topk_rk[i_b, pos].view(-1).tolist()):

                        l_level, l_pos, r_level, r_pos = component_lookup[(pos, n_idx)]

                        num_components = l_level
                        max_possible_trees = index.get_catalan(num_components + 1)
                        if max_possible_trees < lk + 1:
                            is_valid.append(False)
                            continue

                        num_components = r_level
                        max_possible_trees = index.get_catalan(num_components + 1)
                        if max_possible_trees < rk + 1:
                            is_valid.append(False)
                            continue

                        is_valid.append(True)
            is_valid = torch.tensor(is_valid, dtype=torch.bool, device=device).view(B, L, N * K * K, 1)
            all_idx = all_idx[is_valid].view(B, L, -1, 1)
            all_h = all_h[is_valid.expand(B, L, N*K*K, size)].view(B, L, -1, size)

            return all_idx, all_h, topk_h

        def _compose():
            sel_lh = lh.view(B, L, N * K, size).gather(dim=2, index=topk_l.expand(B, L, K, size)).view(-1, size)
            # TODO: If possible, remove this transpose. Although this approach still has less transpose than previously.
            sel_rh = rh.transpose(3, 4).reshape(B, L, N * K, size).gather(dim=2, index=topk_r.expand(B, L, K, size)).view(-1, size)

            topk_h = inside_compose(compose_func, [sel_lh, sel_rh]).view(B, L, K, size)
            topk_h = normalize_func(topk_h)

            return topk_idx, topk_h, topk_h

        all_idx, all_h, topk_h = _dense_compose()
        # other = _compose()
        # check = torch.allclose(topk_h, other, atol=1e-5)
        # assert check, (check, level)

        assert topk_s.shape == (B, L, K, 1), topk_s.shape

        if not force:
            # We should not add more than the number of possible trees to the beam.
            max_possible_trees = index.get_catalan(N + 1)
            if max_possible_trees < K:
                topk_s[:, :, max_possible_trees:] = -1e8

        topk_n_idx = topk_idx // (K**2)
        topk_lk = topk_idx % (K**2) // K
        topk_rk = topk_idx % (K**2) % K

        def verify():
            if not self.debug:
                return

            max_possible_trees = index.get_catalan(N + 1)

            for i_b in range(batch_size):
                for pos in range(L):
                    for i_k in range(K):
                        if max_possible_trees < i_k + 1:
                            continue
                        n_idx = topk_n_idx[i_b, pos, i_k].item()
                        lk = topk_lk[i_b, pos, i_k].item()
                        rk = topk_rk[i_b, pos, i_k].item()
                        l_level, l_pos, r_level, r_pos = component_lookup[(pos, n_idx)]
                        idx = ChartUtil.convert_to_idx(n_idx, lk, rk, N, K)

                        # TODO: Performance implications?
                        if force_info is not None:
                            if idx in force_info['idx_invalid'][(i_b, x_pos)]:
                                continue

                        check_s = s[i_b, pos]
                        check_xs = combo_s[i_b, pos]
                        check_ls = combo_ls[i_b, pos]
                        check_rs = combo_rs[i_b, pos]
                        check_topk_s = topk_s[i_b, pos]

                        check = (check_s, check_topk_s, check_xs, check_ls, check_rs)

                        if l_level == 0:
                            assert lk == 0, (level, pos, i_k, l_level, l_pos, lk, r_level, r_pos, rk, check)
                        if r_level == 0:
                            assert rk == 0, (level, pos, i_k, l_level, l_pos, lk, r_level, r_pos, rk, check)

        verify()

        contains = None
        if force and correction_spans is not None:
            contains = {}
            for i_b in range(batch_size):
                for x_pos in range(L):
                    # If this span is crossing, then no need to apply constraints.
                    if is_crossing(i_b, level, x_pos):
                        continue

                    for i_k in range(K):
                        n_idx = topk_n_idx[i_b, x_pos, i_k].item()
                        l_k = topk_lk[i_b, x_pos, i_k].item()
                        r_k = topk_rk[i_b, x_pos, i_k].item()
                        l_level, l_pos, r_level, r_pos = component_lookup[(x_pos, n_idx)]
                        l_size = l_level + 1
                        r_size = r_level + 1

                        found = False
                        if (l_pos, l_size) in correction_spans[i_b]:
                            if not is_crossing(i_b, r_level, r_pos):
                                found = True
                        elif l_level > 0 and does_contain(i_b, l_level, l_pos, l_k):
                            if not is_crossing(i_b, r_level, r_pos):
                                found = True
                        if (r_pos, r_size) in correction_spans[i_b]:
                            if not is_crossing(i_b, l_level, l_pos):
                                found = True
                        elif r_level > 0 and does_contain(i_b, r_level, r_pos, r_k):
                            if not is_crossing(i_b, l_level, l_pos):
                                found = True
                        if found:
                            contains[(i_b, level, x_pos, i_k)] = True

        # Result.
        result = {}
        for i in range(K):
            result.setdefault('h', []).append(topk_h[:, :, i])
            result.setdefault('s', []).append(topk_s[:, :, i])

            if loss_augmented:
                result.setdefault('inside_error', []).append(topk_error[:, :, i])

        topk_s_local = local_s.gather(index=topk_idx, dim=2)

        result['topk_h'] = topk_h
        result['topk_s'] = topk_s
        result['topk_s_local'] = topk_s_local
        result['topk_n_idx'] = topk_n_idx
        result['topk_lk'] = topk_lk
        result['topk_rk'] = topk_rk

        result['contains'] = contains # Used by following steps.

        # TODO: Can we avoid this?
        result['fake_h'] = lh[:, :, :, 0] # lh : (B, L, N, K, D)
        result['fake_s'] = s.view(B, L, N, K * K, 1)[:, :, :, 0] # s : (B, L, N*K*K ,1)

        return result

    def f_hard_outside_helper(self, inputs, cfg, batch_info):
        device = inputs['device']
        B = inputs['batch_size']
        L = inputs['length'] - cfg['level']
        N = inputs['length'] - cfg['level'] - 1
        size = inputs['size']
        level = cfg['level']
        batch_size = inputs['batch_size']
        length = inputs['length']
        training = inputs['training']

        K = cfg['topk']
        K_constraint = cfg.get('topk-constraint', K)
        force = cfg.get('force', False)
        correction = cfg.get('correction', None)
        correction_spans = cfg.get('correction_spans', None)

        assert correction_spans is None, "Outside does not support span constraints. Edge constraints only."
        if force:
            assert correction is not None

        index = inputs['index']
        icharts = inputs['inside_charts']
        ocharts = inputs['outside_charts']
        CH = {}
        compose_func = inputs['compose_func']
        lse = inputs['lse']
        normalize_func = inputs['normalize_func']
        sibling_dropout_dist = inputs['sibling_dropout_dist']
        coordinate_embeddings_size = inputs['coordinate_embeddings_size']
        project_coordinate_embedding = inputs['project_coordinate_embedding']

        # Incorporate elmo.
        offset_cache = index.get_offset(length)
        components = get_outside_components(length, level, offset_cache)

        component_lookup = {}
        for p_k in range(K):
            for s_k in range(K):
                for i, (p_span, s_span) in enumerate(components):
                    p_level, p_pos = p_span
                    s_level, s_pos = s_span
                    idx = i // L
                    x_pos = i % L
                    component_lookup[(x_pos, idx, p_k, s_k)] = (p_level, p_pos, s_level, s_pos)

        # DIORA.

        ## 0

        p_prod, s_prod, pairs = ChartUtil.get_tensor_product_mask(K)
        p_index, p_info, s_index, s_info = index.get_topk_outside_index(length, level, K)

        def get_outside_states(batch_info, pchart, schart, index, size):

            ps = pchart.index_select(index=p_index, dim=1).view(-1, size)
            ss = schart.index_select(index=s_index, dim=1).view(-1, size)

            return ps, ss

        for i in range(K):
            ph, sh = get_outside_states(batch_info, ocharts[i]['outside_h'], icharts[i]['inside_h'], index, size)
            CH.setdefault('ph', []).append(ph)
            CH.setdefault('sh', []).append(sh)

            ps, ss = get_outside_states(batch_info, ocharts[i]['outside_s'], icharts[i]['inside_s'], index, 1)
            CH.setdefault('ps', []).append(ps)
            CH.setdefault('ss', []).append(ss)

        ## All Combos

        mat = inputs['score_func'].mat

        ph = torch.cat([CH['ph'][i].view(B, L, N, 1, size) for i in range(K)], 3)
        sh = torch.cat([CH['sh'][i].view(B, L, N, size, 1) for i in range(K)], 4)
        if sibling_dropout_dist is not None and self.training:
            mask = sibling_dropout_dist.sample((B, L, N)).tolist()
            mask = torch.tensor(mask, dtype=torch.float, device=device)
            mask = mask.view(B, L, N, 1, 1).expand(B, L, N, size, K).detach()

            if project_coordinate_embedding is not None:
                def get_level(x):
                    return x[2]
                s_size = torch.tensor([get_level(x) + 1 for x in s_info], dtype=torch.long, device=device)
                sibling_replacement = normalize_func(
                    project_coordinate_embedding(
                        coordinate_embeddings_size(s_size)))
                sibling_replacement = sibling_replacement.view(1, L, N, size, 1).expand(B, L, N, size, K)
                sh = torch.where(mask == 0, sibling_replacement, sh)
            else:
                sh = mask * sh
        s_raw = torch.matmul(torch.matmul(ph, mat), sh)
        if lse:
            s_raw = torch.nn.functional.logsigmoid(s_raw)

        select_ps = torch.tensor(p_prod, dtype=torch.float, device=device)
        select_ss = torch.tensor(s_prod, dtype=torch.float, device=device)

        ps = torch.cat([CH['ps'][i].view(B, L, N, 1) for i in range(K)], 3)
        ss = torch.cat([CH['ss'][i].view(B, L, N, 1) for i in range(K)], 3)
        combo_ps = torch.matmul(ps, select_ps).view(B, L, N * K * K, 1)
        combo_ss = torch.matmul(ss, select_ss).view(B, L, N * K * K, 1)
        combo_s = s_raw.view(B, L, N * K * K, 1)

        def logsumexp(a, b, c, dim=1):
            v = torch.cat([a.unsqueeze(dim), b.unsqueeze(dim), c.unsqueeze(dim)], dim)
            d = torch.max(v, dim=dim)[0]
            out = torch.log(torch.exp(v - d.unsqueeze(dim)).sum(dim)) + d
            out[(b < -1e5) | (c < -1e5)] = -1e8
            return out

        if lse:
            s = logsumexp(combo_s, combo_ps, combo_ss, dim=1)
        else:
            s = combo_s + combo_ps + combo_ss

        s_reshape = s
        assert s_reshape.shape == (B, L, N * K * K, 1)

        # We should not include any split that includes an in-complete beam.
        def penalize_incomplete_splits():
            force_index = collections.defaultdict(list)
            for i_b in range(B):
                for pos in range(L):
                    for n_idx in range(N):
                        for p_k, s_k in pairs:
                            new_idx = ChartUtil.outside_convert_to_idx(n_idx, p_k, s_k, N, K)
                            p_level, p_pos, s_level, s_pos = component_lookup[(pos, n_idx, p_k, s_k)]
                            p_num_components = length - p_level - 1
                            s_num_components = s_level
                            max_possible_trees = index.get_catalan(p_num_components + 1)
                            if max_possible_trees < p_k + 1:
                                force_index['i_b'].append(i_b)
                                force_index['pos'].append(pos)
                                force_index['new_idx'].append(new_idx)

                            max_possible_trees = index.get_catalan(s_num_components + 1)
                            if max_possible_trees < s_k + 1:
                                force_index['i_b'].append(i_b)
                                force_index['pos'].append(pos)
                                force_index['new_idx'].append(new_idx)

            s[force_index['i_b'], force_index['pos'], force_index['new_idx']] = -1e8

        if not force:
            # Don't do this when the trees are specified since it's possible for duplicate
            # sub-trees to exist.
            penalize_incomplete_splits()

        # Aggregate.
        topk_s, topk_idx = s_reshape.topk(dim=2, k=K)

        assert topk_s.shape == (B, L, K, 1)
        assert topk_idx.shape == (B, L, K, 1)

        # Force topk according to given constraints.
        if force and correction is not None:
            force_index = collections.defaultdict(list)

            max_possible_trees = index.get_catalan(length)

            assert 0 in correction, "Make sure correction is keyed with levels."

            if level in correction:
                for k, v in correction[level]:
                    i_b, level, pos, i_k = k

                    # Only check for up to K constraints.
                    if i_k + 1 > K_constraint:
                        continue

                    n_idx, p_level, p_pos, p_k, s_level, s_pos, s_k = v

                    new_idx = ChartUtil.outside_convert_to_idx(n_idx, p_k, s_k, N, K)
                    # topk_idx[i_b, pos, i_k] = new_idx
                    force_index['i_b'].append(i_b)
                    force_index['pos'].append(pos)
                    force_index['i_k'].append(i_k)
                    force_index['new_idx'].append(new_idx)

                    if max_possible_trees > i_k:
                        val = combo_ss[i_b, pos, new_idx].item()
                        assert val > -1e5, (val, s_level, s_pos, s_k)
                        val = combo_ps[i_b, pos, new_idx].item()
                        assert val > -1e5, (val, p_level, p_pos, p_k)
                        val = s_reshape[i_b, pos, new_idx].item()
                        assert val > -1e5, (val, level, pos, i_k)

            local_new_idx = torch.tensor(force_index['new_idx'], dtype=torch.long, device=device).unsqueeze(1)
            topk_idx[force_index['i_b'], force_index['pos'], force_index['i_k']] = local_new_idx

            topk_s = s_reshape.gather(index=topk_idx, dim=2)

        # Selective compose.
        topk_n_idx = topk_idx // (K**2)
        topk_pk = topk_idx % (K**2) // K
        topk_sk = topk_idx % (K**2) % K
        # Also can be computed as:
        # n_idx = topk_idx // (K * K) # changes once per K**2 steps
        # p_k = topk_idx // K % K # changes once per K steps
        # s_k = topk_idx % K # changes every step

        topk_p_idx = topk_n_idx * K + topk_pk
        sel_ph = ph.view(B, L, N * K, size).gather(dim=2, index=topk_p_idx.expand(B, L, K, size)).view(-1, size)

        topk_s_idx = topk_n_idx * K + topk_sk
        # TODO: If possible, remove this transpose. Although this approach still has less transpose than previously.
        sel_sh = sh.transpose(3, 4).reshape(B, L, N * K, size).gather(dim=2, index=topk_s_idx.expand(B, L, K, size)).view(-1, size)

        topk_h = outside_compose(compose_func, [sel_ph, sel_sh]).view(B, L, K, size)
        topk_h = normalize_func(topk_h)

        if not force:
            # We should not add more than the number of possible trees to the beam.
            max_possible_trees = index.get_catalan(N + 1)
            if max_possible_trees < K:
                topk_s[:, :, max_possible_trees:] = -1e8

        # Result.
        result = {}
        assert topk_h.shape[2] == K
        assert topk_s.shape[2] == K
        for i_k in range(K):
            result.setdefault('h', []).append(topk_h[:, :, i_k])
            result.setdefault('s', []).append(topk_s[:, :, i_k])

        result['topk_h'] = topk_h
        result['topk_s'] = topk_s
        result['topk_n_idx'] = topk_n_idx
        result['topk_pk'] = topk_pk
        result['topk_sk'] = topk_sk

        # TODO: Can we avoid this?
        assert ph.shape == (B, L, N, K, size)
        assert ps.shape == (B, L, N, K)
        fake_h = ph[:, :, :, 0].transpose(1, 2)
        fake_s = ps[:, :, :, 0].transpose(1, 2).unsqueeze(-1)

        result['fake_h'] = fake_h
        result['fake_s'] = fake_s

        return result


class DioraMLPWithTopk(DioraMLP):
    def __init__(self, *args, **kwargs):
        kwargs = kwargs.copy()
        self.greedy = kwargs.get('greedy', True)
        self.share = kwargs.get('share', False)
        self.dense_compose = kwargs.get('dense_compose', False)
        self.labeled = kwargs.get('labeled', False)
        self.lse = kwargs.get('lse', False)
        sibling_dropout = kwargs.get('sibling_dropout', None)
        coordinate_embeddings = kwargs.get('coordinate_embeddings', False)
        self.charts = None
        super(DioraMLPWithTopk, self).__init__(*args, **kwargs)
        if sibling_dropout is not None:
            self.sibling_dropout_dist = torch.distributions.bernoulli.Bernoulli(torch.FloatTensor([1 - sibling_dropout]))
        else:
            self.sibling_dropout_dist = None
        if coordinate_embeddings:
            self.coordinate_embeddings_size = nn.Embedding(num_embeddings=1000, embedding_dim=16)
            self.project_coordinate_embedding = nn.Linear(16, self.size)
        else:
            self.coordinate_embeddings_size = None
            self.project_coordinate_embedding = None
        self.K = kwargs.get('K', 2)

    def private_inside_hook(self, level, h, s, p, x_s, l_s, r_s):
        pass

    def reset(self):
        super(DioraMLPWithTopk, self).reset()
        if self.charts is not None:
            for i in range(1, self.K):
                chart = self.charts[i]
                keys = list(chart.keys())
                for k in keys:
                    self.nested_del(chart, k)
        self.charts = None

    def safe_set_K(self, val):
        self.reset()
        self.K = val

    def init_parameters(self):
        # Model parameters for transformation required at both input and output
        self.inside_score_func = Bilinear(self.size)
        self.inside_compose_func = ComposeMLP(self.size, self.activation, n_layers=self.n_layers, leaf=True)

        if self.share:
            self.outside_score_func = self.inside_score_func
            self.outside_compose_func = self.inside_compose_func
        else:
            self.outside_score_func = Bilinear(self.size)
            self.outside_compose_func = ComposeMLP(self.size, self.activation, n_layers=self.n_layers)

        self.root_vector_out_h = nn.Parameter(torch.FloatTensor(self.size))
        self.root_vector_out_c = None

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

    def init_with_batch(self, *args, **kwargs):
        result = super(DioraMLPWithTopk, self).init_with_batch(*args, **kwargs)
        self.cache['inside_chart_output'] = dict(by_level={i: {} for i in range(self.batch_size)})
        self.cache['outside_chart_output'] = dict(by_level={i: {} for i in range(self.batch_size)})
        self.cache['topk_s'] = {i: {} for i in range(self.length)}
        self.cache['topk_n_idx'] = {i: {} for i in range(self.length)}
        self.cache['topk_lk'] = {i: {} for i in range(self.length)}
        self.cache['topk_rk'] = {i: {} for i in range(self.length)}
        self.cache['outside_topk_n_idx'] = {i: {} for i in range(self.length)}
        self.cache['outside_topk_pk'] = {i: {} for i in range(self.length)}
        self.cache['outside_topk_sk'] = {i: {} for i in range(self.length)}
        self.cache['cky'] = {i: {} for i in range(self.batch_size)}
        self.cache['saved_trees'] = None
        return result

    def post_inside_hook(self):
        charts = self.charts
        index = self.index
        inside_tree_edges = self.cache['inside_tree_edges']
        batch_size = self.batch_size
        length = self.length
        K = self.K
        root_level, root_pos = length - 1, 0
        tree_edges = {}
        for i_b in range(self.batch_size):
            for i_k in range(self.K):
                tree_edges[(i_b, i_k)] = inside_tree_edges[(i_b, i_k)][(root_level, root_pos)]
        ChartUtil.verify_inside_trees(charts, index, tree_edges, batch_size, length, K)

    def inside_func(self, compose_func, score_func, batch_info, chart, index, normalize_func):
        return self.inside_func_greedy(compose_func, score_func, batch_info, chart, index, normalize_func)

    def outside_func(self, compose_func, score_func, batch_info, chart, index, normalize_func):
        return self.outside_func_greedy(compose_func, score_func, batch_info, chart, index, normalize_func)

    def inside_func_greedy(self, compose_func, score_func, batch_info, chart, index, normalize_func):
        device = self.device
        B = batch_info.batch_size
        L = batch_info.length - batch_info.level
        N = batch_info.level
        level = batch_info.level
        K = self.K

        parameters, inputs, cfg = get_inside_chart_cfg(
            self, new_chart=False, level=level, K=K, device=device,
            compose_func=compose_func, normalize_func=normalize_func, # kwargs
            )

        if self.loss_augmented:
            inputs['loss_augmented'] = True
            inputs['gold_spans'] = self.cache['gold_spans']

        chart_output = ChartUtil(parameters=parameters).run(inputs=inputs, cfg=cfg)['by_level'][level]

        h_0 = chart_output['h'][0]
        s_0 = chart_output['s'][0]

        topk_h = chart_output['topk_h']
        topk_s = chart_output['topk_s']
        topk_s_local = chart_output['topk_s_local']
        topk_n_idx = chart_output['topk_n_idx']
        topk_lk = chart_output['topk_lk']
        topk_rk = chart_output['topk_rk']

        fake_h = chart_output['fake_h']
        fake_s = chart_output['fake_s']

        #

        for i in range(self.K):
            inside_fill_chart(batch_info, self.charts[i], index, chart_output['h'][i], chart_output['s'][i])

            if self.loss_augmented:
                inside_fill_chart(batch_info, self.charts[i], index, error=chart_output['inside_error'][i])

        # Book-keeping.
        offset_cache = index.get_offset(batch_info.length)
        components = get_inside_components(batch_info.length, batch_info.level, offset_cache)

        _, _, pairs = ChartUtil.get_tensor_product_mask(K)
        component_lookup = {}
        for idx, (_, _, x_span, l_span, r_span) in enumerate(components):
            for j, (x_level, x_pos) in enumerate(x_span):
                l_level, l_pos = l_span[j]
                r_level, r_pos = r_span[j]
                component_lookup[(x_pos, idx)] = (l_level, l_pos, r_level, r_pos)

        for pos in range(L):
            self.cache['topk_s'][level][pos] = topk_s[:, pos]
            self.cache['topk_n_idx'][level][pos] = topk_n_idx[:, pos]
            self.cache['topk_lk'][level][pos] = topk_lk[:, pos]
            self.cache['topk_rk'][level][pos] = topk_rk[:, pos]
        self.cache['inside_chart_output']['by_level'][level] = topk_s_local

        # backtrack
        for i_b in range(self.batch_size):
            for pos in range(L):
                for i_k in range(K):
                    n_idx = topk_n_idx[i_b, pos, i_k].item()
                    l_k = topk_lk[i_b, pos, i_k].item()
                    r_k = topk_rk[i_b, pos, i_k].item()
                    l_level, l_pos, r_level, r_pos = component_lookup[(pos, n_idx)]
                    self.cache['inside_tree'][(i_b, i_k)][(level, pos)] = \
                        self.cache['inside_tree'][(i_b, l_k)][(l_level, l_pos)] + \
                        self.cache['inside_tree'][(i_b, r_k)][(r_level, r_pos)] + \
                        [(level, pos)]

                    self.cache['inside_tree_edges'][(i_b, i_k)][(level, pos)] = \
                        self.cache['inside_tree_edges'][(i_b, l_k)][(l_level, l_pos)] + \
                        self.cache['inside_tree_edges'][(i_b, r_k)][(r_level, r_pos)] + \
                        [(level, pos, i_k, l_level, l_pos, l_k, r_level, r_pos, r_k)]

        # HACK: This is inaccurate, but these values are not used so it is okay.
        s = fake_s
        p = s
        h = fake_h

        return h, s, p, None, None, None

    def outside_func_greedy(self, compose_func, score_func, batch_info, chart, index, normalize_func):
        device = self.device
        B = batch_info.batch_size
        L = batch_info.length - batch_info.level
        N = batch_info.length - batch_info.level - 1
        level = batch_info.level
        K = self.K

        parameters, inputs, cfg = get_outside_chart_cfg(
            self, self.charts, new_chart=False, level=level, K=K, device=device,
            compose_func=compose_func, normalize_func=normalize_func, # kwargs
            )

        chart_output = ChartUtil(parameters=parameters).run(inputs=inputs, cfg=cfg)['by_level'][level]

        topk_n_idx = chart_output['topk_n_idx']
        topk_pk = chart_output['topk_pk']
        topk_sk = chart_output['topk_sk']

        fake_h = chart_output['fake_h']
        fake_s = chart_output['fake_s']

        #

        for i in range(K):
            outside_fill_chart(batch_info, self.charts[i], index, chart_output['h'][i], chart_output['s'][i])

        # Book-keeping.

        for pos in range(L):
            self.cache['outside_topk_n_idx'][level][pos] = topk_n_idx[:, pos]
            self.cache['outside_topk_pk'][level][pos] = topk_pk[:, pos]
            self.cache['outside_topk_sk'][level][pos] = topk_sk[:, pos]

        if level == 0:
            inputs['inside_tree'] = self.cache['inside_tree']
            inputs['outside_topk_n_idx'] = self.cache['outside_topk_n_idx']
            inputs['outside_topk_pk'] = self.cache['outside_topk_pk']
            inputs['outside_topk_sk'] = self.cache['outside_topk_sk']
            del cfg['level']
            saved_trees = ChartUtil(parameters=parameters).run_outside_backtrack(inputs=inputs, cfg=cfg)
            self.cache['saved_trees'] = saved_trees

        # HACK: This is inaccurate, but these values are not used so it is okay.
        s = fake_s
        p = s
        h = fake_h

        return h, s, p, None, None, None

    def initialize(self, x):
        result = super(DioraMLPWithTopk, self).initialize(x)
        batch_size, length = x.shape[:2]
        size = self.size
        charts = [self.chart]
        for _ in range(1, self.K):
            charts.append(build_chart(batch_size, length, size, dtype=torch.float, cuda=self.is_cuda))
        # Initialize outside root.
        h = self.root_vector_out_h.view(1, 1, size).expand(batch_size, 1, size)
        h = self.outside_normalize_func(h)
        for i in range(0, self.K):
            if i == 0:
                continue
            # Never should be selected.
            charts[i]['inside_s'][:] = -1e8
            charts[i]['outside_s'][:] = -1e8
            charts[i]['outside_h'][:, -1:] = h
        self.charts = charts
        return result

    def leaf_transform(self, x):
        normalize_func = self.inside_normalize_func
        transform_func = self.inside_compose_func.leaf_transform

        input_shape = x.shape[:-1]
        h = transform_func(x)
        batch_size, length = h.shape[:2]
        h = normalize_func(h.view(*input_shape, self.size))

        return h

