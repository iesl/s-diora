import json

import torch


class ParsePredictor(object):
    def __init__(self, net, word2idx, add_bos_token=False, add_eos_token=False, initial_scalar=1):
        super(ParsePredictor, self).__init__()
        self.net = net
        self.idx2word = {v: k for k, v in word2idx.items()}
        assert add_bos_token is False and add_eos_token is False
        self.initial_scalar = initial_scalar

    def predict(self, batch_map, return_components=False):
        batch = batch_map['sentences']
        example_ids = batch_map['example_ids']

        batch_size = self.net.batch_size

        trees, components = self.parse_batch(batch)

        out = []
        for i in range(batch_size):
            assert trees[i] is not None
            out.append(dict(example_id=example_ids[i], binary_tree=trees[i]))

        if return_components:
            return (out, components)

        return out

    def parse_batch(self, batch, cell_loss=False, return_components=False):
        batch_size = self.net.batch_size
        length = self.net.length
        scalars = self.net.cache['inside_s_components'].copy()
        device = self.net.device
        dtype = torch.float32

        # Assign missing scalars
        for i in range(length):
            scalars[0][i] = torch.full((batch_size, 1), self.initial_scalar, dtype=dtype, device=device)

        leaves = [None for _ in range(batch_size)]
        for i in range(batch_size):
            batch_i = batch[i].tolist()
            leaves[i] = [self.idx2word[idx] for idx in batch_i]

        trees, components = self.batched_cky(scalars, leaves)

        return trees, components

    def batched_cky(self, scalars, leaves):
        batch_size = self.net.batch_size
        length = self.net.length
        device = self.net.device
        dtype = torch.float32

        # Chart.
        chart = [torch.full((length-i, batch_size), 1, dtype=dtype, device=device) for i in range(length)]
        components = {}

        # Backpointers.
        bp = {}
        for ib in range(batch_size):
            bp[ib] = [[None] * (length - i) for i in range(length)]
            bp[ib][0] = [i for i in range(length)]

        for level in range(1, length):
            L = length - level
            N = level

            for pos in range(L):

                pairs, lps, rps, sps = [], [], [], []

                # Assumes that the bottom-left most leaf is in the first constituent.
                spbatch = scalars[level][pos]

                for idx in range(N):
                    # (level, pos)
                    l_level = idx
                    l_pos = pos
                    r_level = level-idx-1
                    r_pos = pos+idx+1

                    # assert l_level >= 0
                    # assert l_pos >= 0
                    # assert r_level >= 0
                    # assert r_pos >= 0

                    l = (l_level, l_pos)
                    r = (r_level, r_pos)

                    lp = chart[l_level][l_pos].view(-1, 1)
                    rp = chart[r_level][r_pos].view(-1, 1)
                    sp = spbatch[:, idx].view(-1, 1)

                    lps.append(lp)
                    rps.append(rp)
                    sps.append(sp)

                    pairs.append((l, r))

                lps, rps, sps = torch.cat(lps, 1), torch.cat(rps, 1), torch.cat(sps, 1)

                ps = lps + rps + sps
                components[(level, pos)] = ps
                argmax = ps.argmax(1).long()

                valmax = ps[range(batch_size), argmax]
                chart[level][pos, :] = valmax

                for i, ix in enumerate(argmax.tolist()):
                    bp[i][level][pos] = pairs[ix]

        trees = []
        for i in range(batch_size):
            tree = self.follow_backpointers(bp[i], leaves[i], bp[i][-1][0])
            trees.append(tree)

        return trees, components

    def follow_backpointers(self, bp, words, pair):
        if isinstance(pair, int):
            return words[pair]

        l, r = pair
        lout = self.follow_backpointers(bp, words, bp[l[0]][l[1]])
        rout = self.follow_backpointers(bp, words, bp[r[0]][r[1]])

        return (lout, rout)
