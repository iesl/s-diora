import os
import json

from collections import Counter

from tqdm import tqdm

import nltk
from nltk.tree import Tree

from experiment_logger import get_logger


WSJ_POS_MAPPING = {'#': 0, '$': 1, "''": 2, ',': 3, '-LRB-': 4, '-RRB-': 5, '.': 6, ':': 7, 'CC': 8, 'CD': 9,
    'DT': 10, 'EX': 11, 'FW': 12, 'IN': 13, 'JJ': 14, 'JJR': 15, 'JJS': 16, 'LS': 17, 'MD': 18, 'NN': 19,
    'NNP': 20, 'NNPS': 21, 'NNS': 22, 'PDT': 23, 'POS': 24, 'PRP': 25, 'PRP$': 26, 'RB': 27, 'RBR': 28, 'RBS': 29,
    'RP': 30, 'SYM': 31, 'TO': 32, 'UH': 33, 'VB': 34, 'VBD': 35, 'VBG': 36, 'VBN': 37, 'VBP': 38, 'VBZ': 39,
    'WDT': 40, 'WP': 41, 'WP$': 42, 'WRB': 43, '``': 44} # 45 Tags

WSJ_CONSTITUENCY_MAPPING = {'#': 0, '$': 1, "''": 2, ',': 3, '-LRB-': 4, '-RRB-': 5, '.': 6, ':': 7, 'ADJP': 8,
    'ADVP': 9, 'CC': 10, 'CD': 11, 'CONJP': 12, 'DT': 13, 'EX': 14, 'FRAG': 15, 'FW': 16, 'IN': 17, 'INTJ': 18,
    'JJ': 19, 'JJR': 20, 'JJS': 21, 'LS': 22, 'LST': 23, 'MD': 24, 'NAC': 25, 'NN': 26, 'NNP': 27, 'NNPS': 28, 'NNS': 29,
    'NP': 30, 'NX': 31, 'PDT': 32, 'POS': 33, 'PP': 34, 'PRN': 35, 'PRP': 36, 'PRP$': 37, 'PRT': 38, 'QP': 39,
    'RB': 40, 'RBR': 41, 'RBS': 42, 'RP': 43, 'RRC': 44, 'S': 45, 'SBAR': 46, 'SBARQ': 47, 'SINV': 48, 'SQ': 49,
    'SYM': 50, 'TO': 51, 'UCP': 52, 'UH': 53, 'VB': 54, 'VBD': 55, 'VBG': 56, 'VBN': 57, 'VBP': 58, 'VBZ': 59, 'VP': 60,
    'WDT': 61, 'WHADJP': 62, 'WHADVP': 63, 'WHNP': 64, 'WHPP': 65, 'WP': 66, 'WP$': 67, 'WRB': 68, 'X': 69, '``': 70}

COMPOUND_TAGS = [('ADJP', 'FRAG'), ('ADJP', 'NP'), ('ADJP', 'QP'), ('ADJP', 'S'), ('ADVP', 'FRAG'), ('ADVP', 'S'), ('ADVP', 'VP'),
    ('FRAG', 'NP'), ('FRAG', 'PP'), ('FRAG', 'SBAR'), ('FRAG', 'VP'), ('FRAG', 'WHADVP'), ('NP', 'QP'), ('NP', 'S'),
    ('NP', 'S', 'SBAR', 'VP'), ('NP', 'S', 'VP'), ('NP', 'SBAR'), ('NP', 'X'), ('PP', 'S'), ('PP', 'VP'), ('PP', 'X'),
    ('PRN', 'S'), ('PRN', 'SINV'), ('S', 'SBAR'), ('S', 'SBAR', 'VP'), ('S', 'UCP'), ('S', 'VP'), ('SBAR', 'SINV'), ('SQ', 'VP')]
for t in COMPOUND_TAGS:
    WSJ_CONSTITUENCY_MAPPING[t] = len(WSJ_CONSTITUENCY_MAPPING)

# To support binary trees.
WSJ_CONSTITUENCY_MAPPING['-NONE-POS-'] = len(WSJ_CONSTITUENCY_MAPPING)
WSJ_CONSTITUENCY_MAPPING['-NONE-'] = len(WSJ_CONSTITUENCY_MAPPING)

INVERSE_POS_MAPPING = {v: k for k, v in WSJ_POS_MAPPING.items()}
INVERSE_CONSTITUENCY_MAPPING = {v: k for k, v in WSJ_CONSTITUENCY_MAPPING.items()}

WSJ_CONSTITUENCY_MAPPING_WITHOUT_POS = {k: v for k, v in WSJ_CONSTITUENCY_MAPPING.items() if k not in WSJ_POS_MAPPING}
INVERSE_CONSTITUENCY_MAPPING_WITHOUT_POS = {v: k for k, v in WSJ_CONSTITUENCY_MAPPING_WITHOUT_POS.items()}
WSJ_CONSTITUENCY_MAPPING_ONLY_POS = {k: v for k, v in WSJ_CONSTITUENCY_MAPPING.items() if k in WSJ_POS_MAPPING}
INVERSE_CONSTITUENCY_MAPPING_ONLY_POS = {v: k for k, v in WSJ_CONSTITUENCY_MAPPING_ONLY_POS.items()}
# IGNORED_TAGS = (",", ":", "``", "''", ".")


class Node(object):
    def __init__(self, parse, label, pos=None, size=None):
        super(Node, self).__init__()
        self.parse = parse
        self.label = label
        self.pos = pos
        self.size = size

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '({}, {})'.format(self.parse, self.label)


def tree_to_spans(tree, minsize=1, skip_root=False):
    spans = []
    span_children = []

    def helper(tr, pos):
        if not isinstance(tr, (list, tuple)):
            size = 1
            if size >= minsize:
                spans.append((pos, size))
                span_children.append([])
            return size

        if len(tr) == 1:
            return tr[0]

        size = 0
        children = []
        for x in tr:
            xpos = pos + size
            xsize = helper(x, xpos)
            size += xsize
            children.append((xpos, xsize))
        if size >= minsize:
            spans.append((pos, size))
            span_children.append(children)

        return size

    helper(tree, 0)

    if skip_root:
        spans.pop()
        span_children.pop()

    assert len(spans) == len(span_children)

    return spans, span_children


def add_null_spans(span_lst, binary_tree):
    binary_tree_spans = set(tree_to_spans(binary_tree)[0])
    existing_spans = set([(pos, size) for pos, size, label in span_lst])
    span_to_label = {}
    for pos, size, label in span_lst:
        k = (pos, size)
        assert k not in span_to_label, (pos, size, label, span_to_label[k])
        span_to_label[k] = label
    for sp in binary_tree_spans:
        if sp not in existing_spans:
            pos, size = sp
            if size == 1:
                span_lst.append((pos, size, WSJ_CONSTITUENCY_MAPPING['-NONE-POS-']))
                #print('[read] add -NONE-POS-')
            else:
                span_lst.append((pos, size, WSJ_CONSTITUENCY_MAPPING['-NONE-']))
        else:
            pos, size = sp
            #print('[read] [single] found {}'.format(INVERSE_CONSTITUENCY_MAPPING[span_to_label[(pos, size)]]))
    return span_lst


def node2entity_labels(node):
    """
    Ignores width 1.
    """
    entity_labels = []

    def func(node, pos=0):
        if isinstance(node.parse, str):
            # if not ignore_unary:
            #     entity_labels.append((pos, 1, node.label))
            return 1

        sofar = 0
        for x in node.parse:
            xsize = func(x, pos + sofar)
            sofar += xsize

        size = sofar
        entity_labels.append((pos, size, node.label))

        return size

    func(node)

    return entity_labels


def get_labels(node, position=0):
    if isinstance(node.parse, str):
        size = 1
        return size, [(position, size, node.label)]

    total_size = 0
    total_lst = []
    for x in node.parse:
        size, lst = get_labels(x, position+total_size)
        total_size += size
        total_lst += lst

    return total_size, total_lst + [(position, total_size, node.label)]


def parse_to_tuples(parse):
    skipped = []
    def helper(parse, label, length, pos=0):
        if isinstance(parse, str):
            x = Node(parse, label, pos=pos, size=1)
            is_pos = True
            return x, is_pos
        is_pos = False

        node_lst = []
        is_pos_lst = []
        size = 0
        for x in parse:
            xnode, xis_pos = helper(x, parse.label(), len(parse), pos=pos+size)
            size += xnode.size
            node_lst.append(xnode)
            is_pos_lst.append(xis_pos)
        result = tuple(node_lst)

        if len(result) == 1: # unary
            _is_pos = is_pos_lst[0]
            if length == 1:
                child_label = result[0].label
                skipped.append((_is_pos, label, child_label, pos, size))
            return result[0], _is_pos
        elif length == 1:
            child_label = parse.label()
            skipped.append((is_pos, label, child_label, pos, size))

        x = Node(result, parse.label(), pos=pos, size=size)
        return x, is_pos
    assert len(parse) == 1
    result, _ = helper(parse[0], label=parse.label(), length=len(parse))

    return result, skipped


def pick(lst, k):
    return [d[k] for d in lst]


def convert_binary_bracketing(parse, lowercase=True):
    transitions = []
    tokens = []

    for word in parse.split(' '):
        if word[0] != "(":
            if word == ")":
                transitions.append(1)
            else:
                # Downcase all words to match GloVe.
                if lowercase:
                    tokens.append(word.lower())
                else:
                    tokens.append(word)
                transitions.append(0)
    return tokens, transitions


def build_tree(tokens, transitions):
    stack = []
    buf = tokens[::-1]

    for t in transitions:
        if t == 0:
            stack.append(buf.pop())
        elif t == 1:
            right = stack.pop()
            left = stack.pop()
            stack.append((left, right))

    assert len(stack) == 1

    return stack[0]


def get_spans_and_siblings(tree):
    def helper(tr, idx=0, name='root'):
        if isinstance(tr, (str, int)):
            return 1, [(idx, idx+1)], []

        l_size, l_spans, l_sibs = helper(tr[0], name='l', idx=idx)
        r_size, r_spans, r_sibs = helper(tr[1], name='r', idx=idx+l_size)

        size = l_size + r_size

        # Siblings.
        spans = [(idx, idx+size)] + l_spans + r_spans
        siblings = [(l_spans[0], r_spans[0], name)] + l_sibs + r_sibs

        return size, spans, siblings

    _, spans, siblings = helper(tree)

    return spans, siblings


def get_spans_with_labels(tree):
    def helper(tr, idx=0):
        if isinstance(tr, (str, int)):
            return 1, []

        spans = []
        sofar = idx

        for subtree in tr:
            size, subspans = helper(subtree, idx=sofar)
            spans += subspans
            sofar += size

        size = sofar - idx
        spans += [(idx, sofar, tr.label())]

        return size, spans

    _, spans = helper(tree)

    return spans


def get_coordinates_with_labels(tree):
    spans = get_spans_with_labels(tree)
    coordinates = [(x[0], x[1]-x[0], x[2]) for x in spans]
    return coordinates


def get_spans(tree):
    def helper(tr, idx=0):
        if isinstance(tr, (str, int)):
            return 1, []

        spans = []
        sofar = idx

        for subtree in tr:
            size, subspans = helper(subtree, idx=sofar)
            spans += subspans
            sofar += size

        size = sofar - idx
        spans += [(idx, sofar)]

        return size, spans

    _, spans = helper(tree)

    return spans


class BaseTextReader(object):
    def __init__(self, lowercase=True, filter_length=0, include_id=False):
        self.lowercase = lowercase
        self.filter_length = filter_length if filter_length is not None else 0
        self.include_id = include_id

    def read(self, filename):
        return self.read_sentences(filename)

    def read_sentences(self, filename):
        sentences = []
        extra = dict()

        if self.include_id:
            extra['example_ids'] = []

        with open(filename) as f:
            for line in tqdm(f, desc='read'):
                for s in self.read_line(line):
                    if self.filter_length > 0 and len(s) > self.filter_length:
                        continue
                    if self.include_id:
                        sid = s[0]
                        s = s[1:]
                        extra['example_ids'].append(sid)
                    sentences.append(s)

        return {
            "sentences": sentences,
            "extra": extra
            }

    def read_line(self, line):
        raise NotImplementedError


class PlainTextReader(BaseTextReader):
    r"""A class for reading files where each line is delimited by a symbol (default is ' ').
    """

    def __init__(self, lowercase=True, filter_length=0, delim=' ', include_label=None, include_id=False):
        super(PlainTextReader, self).__init__(lowercase=lowercase, filter_length=filter_length, include_id=include_id)
        self.delim = delim

    def read_line(self, line):
        example = line.strip().split(self.delim)
        yield example


class WSJEMNLPReader(object):
    def __init__(self, lowercase=True, filter_length=0, for_nopunct=False):
        self.lowercase = lowercase
        self.filter_length = filter_length
        self.for_nopunct = for_nopunct
        self.logger = get_logger()

    def parse_nltk(self, tr):
        result = {}

        do_not_include = ('ROOT',)

        def helper(tr, pos=0):
            if len(tr) == 1 and isinstance(tr[0], str):
                size = 1
                # Don't include part of speech.
                return size
            if len(tr) == 1:
                size = helper(tr[0], pos)
                label = tr.label()
                if label not in do_not_include:
                    result.setdefault((pos, size), []).append(label)
                return size

            size = 0
            for x in tr:
                xsize = helper(x, pos=pos+size)
                size += xsize

            label = tr.label()
            if label not in do_not_include:
                result.setdefault((pos, size), []).append(label)

            return size

        length = helper(tr, 0)

        for pos in range(length):
            if (pos, 1) not in result:
                result[(pos, 1)] = ('-NONE-POS-',)

        return list(result.items())

    def indexify_constituency_tags(self, constituency_tags):
        mapping = {}
        def helper(d):
            for span, label_lst in d:
                assert isinstance(label_lst, (list, tuple))
                pos, size = span
                if len(label_lst) == 0:
                    continue
                for y in label_lst:
                    if y not in WSJ_CONSTITUENCY_MAPPING_WITHOUT_POS:
                        print('WARNING: Bad value for constituency tag.')
                        break
                if len(label_lst) > 1:
                    key = tuple(sorted(label_lst))
                    if key not in WSJ_CONSTITUENCY_MAPPING:
                        new_key = key[-1]
                        print('TRUNCATING KEY {} -> {}'.format(key, new_key))
                        key = new_key
                else:
                    key = label_lst[0]
                label = WSJ_CONSTITUENCY_MAPPING_WITHOUT_POS.get(key, '-NONE-')
                yield (pos, size, label)
        constituency_tags = [list(helper(d)) for d in constituency_tags]
        mapping = {k: i for i, k in enumerate(sorted(mapping.keys()))}
        return constituency_tags

    def indexify_pos_tags(self, pos_tag_lst):
        def helper(seq):
            return [WSJ_POS_MAPPING[tag] for tag in seq]
        pos_tag_lst = [helper(seq) for seq in pos_tag_lst]
        return pos_tag_lst

    def read(self, path):
        sentences = []
        extra = {}

        with open(path) as f:
            for line in tqdm(f):
                data = json.loads(line)
                example_id = data['example_id']
                tokens = data['tokens']

                if self.for_nopunct:
                    if self.filter_length > 0 and len(data['tokens_no_punct']) > self.filter_length:
                        continue
                elif self.filter_length > 0 and len(tokens) > self.filter_length:
                    continue
                nltk_tree = nltk.Tree.fromstring(data['raw_parse'])
                binary_tree = data['binary_tree']
                pos_tags = [x[1] for x in nltk_tree.pos()]
                constituency_tags = self.parse_nltk(nltk_tree)

                if self.lowercase:
                    tokens = [x.lower() for x in tokens]

                # Add to dataset.
                sentences.append(tokens)
                extra.setdefault('example_ids', []).append(example_id)
                extra.setdefault('constituency_tags', []).append(constituency_tags)
                extra.setdefault('pos_tags', []).append(pos_tags)
                extra.setdefault('binary_tree', []).append(binary_tree)
                extra.setdefault('raw_parse', []).append(data['raw_parse'])

                for k in ['labeled_spans', 'overlaps_with_onto']:
                    if k in data:
                        extra.setdefault(k, []).append(data[k])

        constituency_tags = [add_null_spans(x, tr) for x, tr in zip(self.indexify_constituency_tags(extra['constituency_tags']), extra['binary_tree'])]
        pos_tags = self.indexify_pos_tags(extra['pos_tags'])

        extra['constituency_tags'] = constituency_tags
        extra['pos_tags'] = pos_tags

        metadata = {}
        metadata['n_etypes'] = 0
        metadata['etype2idx'] = {}
        metadata['tag2idx'] = WSJ_POS_MAPPING

        self.logger.info('pos tags = {}'.format(WSJ_POS_MAPPING))
        self.logger.info('# of pos tags = {}'.format(len(WSJ_POS_MAPPING)))

        self.logger.info('constituency tags = {}'.format(WSJ_CONSTITUENCY_MAPPING))
        self.logger.info('# of constituency tags = {}'.format(len(WSJ_CONSTITUENCY_MAPPING)))

        return {
            "sentences": sentences,
            "extra": extra,
            "metadata": metadata,
            }

