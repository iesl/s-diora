import torch

from offset_cache import get_offset_cache


class OutsideIndex(object):
    def get_pairs(self, level, i, n):
        """
        Returns all (parent, sibling) coordinate pairs that
        are used to construct a node at coordinates
        (level, i) where there n leaf nodes.

        """
        pairs = []

        for level_ in range(level + 1, i + 1):
            p_level = level_
            p_i = i
            s_level = level_ - level - 1
            s_i = i - level - 1

            pairs.append([(p_level, p_i), (s_level, s_i)])

        for i_ in range(i + 1, n):
            p_level = level + i_ - i
            p_i = i_
            s_level = i_ - i - 1
            s_i = i_

            pairs.append([(p_level, p_i), (s_level, s_i)])

        return pairs

    def xget_all_pairs(self, level, n):
        pairs = []
        for i in range(level, n):
            pairs += self.get_pairs(level, i, n)
        return pairs

    def get_all_pairs(self, level, n):
        L = n - level
        N = L - 1

        pairs = []

        for i in range(N):
            jseen = 0
            for j in range(L):
                if j < N - i:
                    s_level = n - i - 1
                    s_i = N - i - j - 1
                    p_level = s_level
                    p_i = s_level - j
                else:
                    s_level = j - 1
                    s_i = jseen
                    p_level = n - (N - s_level)
                    p_i = n - (N - s_i)
                    jseen += 1
                pair = [(p_i, p_level), (s_i, s_level)]
                pairs.append(pair)

        return pairs

    def get_leaf(self, n):
        node2leaf = {}

        for level in range(1, n):
            L = n - level
            for pos in range(L):
                node2leaf[(level, pos)] = []
                for i in range(L-1):
                    offset = level + 1
                    leaf =  (pos + offset + i) % n
                    node2leaf[(level, pos)].append(leaf)

        return node2leaf


class OutsideIndexCheck(object):
    def __init__(self, index, length, spans, siblings):
        sib_map = {}
        for x, y, n in siblings:
            sib_map[x] = (y, n)
            sib_map[y] = (x, n)

        check = {}
        for sibling, (target, name) in sib_map.items():
            xpos = target[0]
            xsize = target[1] - target[0]
            xlevel = xsize - 1
            xoffset = index.get_offset(length)[xlevel]
            xidx = xoffset + xpos

            spos = sibling[0]
            ssize = sibling[1] - sibling[0]
            parent = (min(xpos, spos), min(xpos, spos) + xsize + ssize)

            ppos = parent[0]
            psize = parent[1] - parent[0]
            plevel = psize - 1
            poffset = index.get_offset(length)[plevel]
            pidx = poffset + ppos

            check[(pidx, xidx)] = True
        self.check = check

    def is_valid(self, pidx, xidx):
        return (pidx, xidx) in self.check


def get_outside_components(length, level, offset_cache=None):
    if offset_cache is None:
        offset_cache = get_offset_cache(length)
    index = OutsideIndex()
    pairs = index.get_all_pairs(level, length)
    output = []

    for pair in pairs:
        par, sis = pair
        par_lvl = par[0]
        par_pos = par[1] - par[0]
        par_span = (par_lvl, par_pos)
        sis_lvl = sis[0]
        sis_pos = sis[1] - sis[0]
        sis_span = (sis_lvl, sis_pos)

        output.append((par_span, sis_span))

    return output


def get_outside_index(length, level, offset_cache=None, cuda=False):
    if offset_cache is None:
        offset_cache = get_offset_cache(length)
    index = OutsideIndex()
    pairs = index.get_all_pairs(level, length)

    par_lvl, par_pos = [], []
    sis_lvl, sis_pos = [], []

    for pair in pairs:
        par, sis = pair
        par_lvl.append(par[0])
        par_pos.append(par[1] - par[0])
        sis_lvl.append(sis[0])
        sis_pos.append(sis[1] - sis[0])

    device = torch.cuda.current_device() if cuda else None

    # Parent
    index = []
    for lvl, pos in zip(par_lvl, par_pos):
        offset = offset_cache[lvl]
        idx = offset + pos
        index.append(idx)
    par_index = torch.tensor(index, dtype=torch.long, device=device)

    # Sibling
    index = []
    for lvl, pos in zip(sis_lvl, sis_pos):
        offset = offset_cache[lvl]
        idx = offset + pos
        index.append(idx)
    sis_index = torch.tensor(index, dtype=torch.long, device=device)

    return par_index, sis_index


def get_topk_outside_index(length, level, K, offset_cache=None, cuda=False):
    if offset_cache is None:
        offset_cache = get_offset_cache(length)

    L = length - level
    N = length - level - 1

    components = get_outside_components(length, level, offset_cache)

    p_info, s_info = [], []
    for i, (p_span, s_span) in enumerate(components):
        p_level, p_pos = p_span
        s_level, s_pos = s_span
        n_idx = i // L
        x_pos = i % L
        p_idx = offset_cache[p_level] + p_pos
        s_idx = offset_cache[s_level] + s_pos

        p_info.append((x_pos, n_idx, p_level, p_pos, p_idx))
        s_info.append((x_pos, n_idx, s_level, s_pos, s_idx))

    def sort_key(x):
        x_pos, n_idx, inp_level, inp_pos, inp_idx = x
        return (x_pos, n_idx)

    def get_val(x):
        x_pos, n_idx, inp_level, inp_pos, inp_idx = x
        return inp_idx

    p_info = sorted(p_info, key=sort_key)
    s_info = sorted(s_info, key=sort_key)

    device = torch.cuda.current_device() if cuda else None

    p_index = torch.tensor([get_val(x) for x in p_info], dtype=torch.long, device=device)
    s_index = torch.tensor([get_val(x) for x in s_info], dtype=torch.long, device=device)

    return p_index, p_info, s_index, s_info


def get_outside_encoded_index(length, offset_cache=None, cuda=False):
    if offset_cache is None:
        offset_cache = get_offset_cache(length)
    index = OutsideIndex()

    outside_leaf = {}
    node2leaf = index.get_leaf(length)

    device = torch.cuda.current_device() if cuda else None

    for (lvl, pos), leaf in node2leaf.items():
        offset = offset_cache[lvl]
        idx = offset + pos
        leaf = torch.tensor(leaf, dtype=torch.long, device=device)
        outside_leaf[idx] = leaf

    return outside_leaf


def get_outside_target(length, level, offset_cache=None, cuda=False):
    if offset_cache is None:
        offset_cache = get_offset_cache(length)

    L = length - level
    offset = offset_cache[level]

    target = []
    for i in range(L-1):
        target.extend(range(offset, offset + L))

    device = torch.cuda.current_device() if cuda else None
    target = torch.tensor(target, dtype=torch.long, device=device)
    return target

