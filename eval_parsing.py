import collections
import json
import os

import nltk
from nltk.treeprettyprinter import TreePrettyPrinter
import numpy as np
import torch
from tqdm import tqdm

from cky import ParsePredictor as CKY
from experiment_logger import get_logger
from evaluation_utils import BaseEvalFunc


def convert_to_nltk(tr, label='|'):
    def helper(tr):
        if not isinstance(tr, (list, tuple)):
            return '({} {})'.format(label, tr)
        nodes = []
        for x in tr:
            nodes.append(helper(x))
        return '({} {})'.format(label, ' '.join(nodes))
    return helper(tr)


def example_f1(gt, pred):
    correct = len(gt.intersection(pred))
    if correct == 0:
        return 0., 0., 0.
    gt_total = len(gt)
    pred_total = len(pred)
    prec = float(correct) / pred_total
    recall = float(correct) / gt_total
    f1 = 2 * (prec * recall) / (prec + recall)
    return f1, prec, recall


def tree_to_spans(tree):
    spans = []

    def helper(tr, pos):
        if not isinstance(tr, (list, tuple)):
            size = 1
            return size
        size = 0
        for x in tr:
            xpos = pos + size
            xsize = helper(x, xpos)
            size += xsize
        spans.append((pos, size))
        return size

    helper(tree, 0)

    return spans


def spans_to_tree(spans, tokens):
    length = len(tokens)

    # Add missing spans.
    span_set = set(spans)
    for pos in range(length):
        if pos not in span_set:
            spans.append((pos, 1))

    spans = sorted(spans, key=lambda x: (x[1], x[0]))

    pos_to_node = {}
    root_node = None

    for i, span in enumerate(spans):

        pos, size = span

        if i < length:
            assert i == pos
            node = (pos, size, tokens[i])
            pos_to_node[pos] = node
            continue

        node = (pos, size, [])

        for i_pos in range(pos, pos+size):
            child = pos_to_node[i_pos]
            c_pos, c_size = child[0], child[1]

            if i_pos == c_pos:
                node[2].append(child)
            pos_to_node[i_pos] = node

    def helper(node):
        pos, size, x = node
        if isinstance(x, str):
            return x
        return tuple([helper(xx) for xx in x])

    root_node = pos_to_node[0]
    tree = helper(root_node)

    return tree


class TreesFromDiora(object):
    def __init__(self, diora, word2idx, outside, oracle):
        self.diora = diora
        self.word2idx = word2idx
        self.idx2word = {idx: w for w, idx in word2idx.items()}
        self.outside = outside
        self.oracle = oracle

    def to_spans(self, lst):
        return [(pos, level + 1) for level, pos in lst]

    def predict(self, batch_map):
        batch_size, length = batch_map['sentences'].shape
        example_ids = batch_map['example_ids']
        tscores = [0.0] * batch_size
        K = self.diora.K

        for i_b in range(batch_size):
            tokens = batch_map['ground_truth'][i_b]['tokens']
            root_level, root_pos = length - 1, 0
            spans = self.to_spans(self.diora.cache['inside_tree'][(i_b, 0)][(root_level, root_pos)])
            binary_tree = spans_to_tree(spans, tokens)
            other_trees = []

            yield dict(example_id=example_ids[i_b], binary_tree=binary_tree, binary_tree_score=tscores[i_b], other_trees=other_trees)


class ParsingComponent(BaseEvalFunc):

    def init_defaults(self):
        self.agg_mode = 'sum'
        self.cky_mode = 'sum'
        self.ground_truth = None
        self.inside_pool = 'sum'
        self.oracle = {'use': False}
        self.outside = True
        self.seed = 121
        self.semi_supervised = False
        self.K = None
        self.choose_tree = 'local'

    def compare(self, prev_best, results):
        out = []
        key, val, is_best = 'placeholder', None, True
        out.append((key, val, is_best))
        return out

    def parse(self, trainer, info):
        logger = self.logger

        multilayer = False
        diora = trainer.get_single_net(trainer.net).diora
        if hasattr(diora, 'layers'):
            multilayer = True
            pred_lst = []
            for i, layer in enumerate(diora.layers):
                logger.info(f'Diora Layer {i}:')
                pred = self.single_layer_parser(trainer, layer, info)
                pred_lst.append(pred)
        else:
            pred_lst = self.single_layer_parser(trainer, diora, info)
        return pred_lst, multilayer

    def single_layer_parser(self, trainer, diora, info):
        logger = self.logger
        epoch = info.get('epoch', 0)

        original_K = diora.K
        if self.K is not None:
            diora.safe_set_K(self.K)

        # set choose_tree
        if hasattr(diora, 'choose_tree'):
            original_choose_tree = diora.choose_tree
            diora.choose_tree = self.choose_tree

        word2idx = self.dataset['word2idx']
        if self.cky_mode == 'cky':
            parse_predictor = CKY(net=diora, word2idx=word2idx,
                add_bos_token=trainer.net.add_bos_token, add_eos_token=trainer.net.add_eos_token)
        elif self.cky_mode == 'diora':
            parse_predictor = TreesFromDiora(diora=diora, word2idx=word2idx, outside=self.outside, oracle=self.oracle)

        batches = self.batch_iterator.get_iterator(random_seed=self.seed, epoch=epoch)

        logger.info('Parsing.')

        pred_lst = []
        counter = 0
        eval_cache = {}

        if self.ground_truth is not None:
            self.ground_truth = os.path.expanduser(self.ground_truth)
            ground_truth_data = {}
            with open(self.ground_truth) as f:
                for line in f:
                    ex = json.loads(line)
                    ground_truth_data[ex['example_id']] = ex

        # Eval loop.
        with torch.no_grad():
            for i, batch_map in enumerate(batches):
                batch_size, length = batch_map['sentences'].shape

                if length <= 2:
                    continue

                example_ids = batch_map['example_ids']
                if self.ground_truth is not None:
                    batch_ground_truth = [ground_truth_data[x] for x in example_ids]
                    batch_map['ground_truth'] = batch_ground_truth

                _ = trainer.step(batch_map, train=False, compute_loss=False, info={ 'inside_pool': self.inside_pool, 'outside': self.outside })

                for j, x in enumerate(parse_predictor.predict(batch_map)):

                    pred_lst.append(x)

                self.eval_loop_hook(trainer, diora, info, eval_cache, batch_map)

        self.post_eval_hook(trainer, diora, info, eval_cache)

        diora.safe_set_K(original_K)

        # set choose_tree
        if hasattr(diora, 'choose_tree'):
            diora.choose_tree = original_choose_tree

        return pred_lst

    def eval_loop_hook(self, trainer, diora, info, eval_cache, batch_map):
        pass

    def post_eval_hook(self, trainer, diora, info, eval_cache):
        pass

    def run(self, trainer, info):
        logger = self.logger
        outfile = info.get('outfile', None)
        pred_lst, multilayer = self.parse(trainer, info)

        if self.write:
            corpus = collections.OrderedDict()

            # Read the ground truth.
            with open(self.ground_truth) as f:
                for line in f:
                    ex = json.loads(line)
                    corpus[ex['example_id']] = ex

            def to_raw_parse(tr):
                def helper(tr):
                    if isinstance(tr, (str, int)):
                        return '(DT {})'.format(tr)
                    nodes = []
                    for x in tr:
                        nodes.append(helper(x))
                    return '(S {})'.format(' '.join(nodes))
                return '(ROOT {})'.format(helper(tr))

            # Write more general format.
            path = outfile + '.pred'
            logger.info('writing parse tree output -> {}'.format(path))
            with open(path, 'w') as f:
                for x in pred_lst:
                    pred_binary_tree = x['binary_tree']
                    f.write(to_raw_parse(pred_binary_tree) + '\n')

            path = outfile + '.gold'
            logger.info('writing parse tree output -> {}'.format(path))
            with open(path, 'w') as f:
                for x in pred_lst:
                    example_id = x['example_id']
                    gt = corpus[example_id]
                    gt_binary_tree = gt['binary_tree']
                    f.write(to_raw_parse(gt_binary_tree) + '\n')

            path = outfile + '.diora'
            logger.info('writing parse tree output -> {}'.format(path))
            with open(path, 'w') as f:
                for x in pred_lst:
                    example_id = x['example_id']
                    gt = corpus[example_id]
                    o = collections.OrderedDict()
                    o['example_id'] = example_id
                    o['binary_tree'] = x['binary_tree']
                    o['raw_parse'] = to_raw_parse(x['binary_tree'])
                    o['tokens'] = gt['tokens']
                    f.write(json.dumps(o) + '\n')

        eval_result = dict()
        eval_result['name'] = self.name
        eval_result['meta'] = dict()

        return eval_result
