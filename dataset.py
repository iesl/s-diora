from collections import deque
import copy
import os

import torch
import numpy as np

from tqdm import tqdm

from reading import *
from batch_iterator import BatchIterator
from embeddings import EmbeddingsReader, UNK_TOKEN
from preprocessing import indexify, indexify_etype, build_text_vocab
from preprocessing import synthesize_training_data
from experiment_logger import get_logger


class ConsolidateDatasets(object):
    """
    A class for consolidating many datasets.
    """

    def __init__(self, datasets):
        self.datasets = datasets
        self.logger = get_logger()

    def reindex(self, sentences, inverse_mapping):
        def fn(s):
            for idx in s:
                yield inverse_mapping[idx]
        def queue(lst):
            q = deque(lst)
            while len(q) > 0:
                yield q.popleft()
        return [list(fn(s)) for s in tqdm(queue(sentences), desc='reindex')]

    def remap_embeddings(self, datasets, inverse_mapping_lst, master_word2idx):
        size = datasets[0]['embeddings'].shape[1]
        embeddings = np.zeros((len(master_word2idx), size), dtype=np.float32)
        for dset, old2master in zip(datasets, inverse_mapping_lst):
            idx_from, idx_to = zip(*old2master.items())
            embeddings[np.asarray(idx_to)] = dset['embeddings'][np.asarray(idx_from)]

        return embeddings

    def consolidate_word2idx(self, word2idx_lst):
        master_word2idx = {}
        inverse_mapping_lst = []

        for w2i in word2idx_lst:
            old2master = {}
            for w, idx in w2i.items():
                if w not in master_word2idx:
                    master_word2idx[w] = len(master_word2idx)
                old2master[idx] = master_word2idx[w]
            inverse_mapping_lst.append(old2master)

        return master_word2idx, inverse_mapping_lst

    def consolidate_etype2idx(self, etype2idx_lst):
        master_etype2idx = {}
        inverse_mapping_lst = []

        overlapping_etypes = set()
        for i, e2i in enumerate(etype2idx_lst):
            keys = list(e2i.keys())
            if i == 0:
                overlapping_etypes.update(keys)
                continue
            overlapping_etypes = set.intersection(overlapping_etypes, set(keys))

        self.logger.info('[consolidate] overlapping_etypes = {}'.format(len(overlapping_etypes)))

        for i, e in enumerate(sorted(overlapping_etypes)):
            master_etype2idx[e] = i

        # TODO: Remove.
        master_etype2idx = {e: i for i, e in enumerate(type_lst)}

        for e2i in etype2idx_lst:
            old2master = {}
            for e, idx in e2i.items():
                if e not in master_etype2idx:
                    continue
                old2master[idx] = master_etype2idx[e]
            inverse_mapping_lst.append(old2master)

        return master_etype2idx, inverse_mapping_lst

    def run(self):
        word2idx_lst = [x['word2idx'] for x in self.datasets]
        master_word2idx, inverse_mapping_lst = self.consolidate_word2idx(word2idx_lst)
        embeddings = self.remap_embeddings(self.datasets, inverse_mapping_lst, master_word2idx)
        for dset, inverse_mapping in zip(self.datasets, inverse_mapping_lst):
            dset['sentences'] = self.reindex(dset['sentences'], inverse_mapping)
            dset['word2idx'] = master_word2idx
            dset['embeddings'] = embeddings


class ReaderManager(object):
    def __init__(self, reader):
        super(ReaderManager, self).__init__()
        self.reader = reader
        self.logger = get_logger()

    def run(self, options, text_path, embeddings_path):
        reader = self.reader
        logger = self.logger

        logger.info('Reading text: {}'.format(text_path))
        reader_result = reader.read(text_path)
        sentences = reader_result['sentences']
        extra = reader_result['extra']
        metadata = reader_result.get('metadata', {})
        logger.info('len(sentences)={}'.format(len(sentences)))
        if 'n_etypes' in metadata:
            logger.info('n_etypes={}'.format(metadata['n_etypes']))

        word2idx = build_text_vocab(sentences)
        logger.info('len(vocab)={}'.format(len(word2idx)))

        if 'embeddings' in metadata:
            logger.info('Using embeddings from metadata.')
            embeddings = metadata['embeddings']
            del metadata['embeddings']
        else:
            logger.info('Reading embeddings.')
            embeddings, word2idx = EmbeddingsReader().get_embeddings(
                options, embeddings_path, word2idx)

        unk_index = word2idx.get(UNK_TOKEN, None)
        logger.info('Converting tokens to indexes (unk_index={}).'.format(unk_index))
        sentences = indexify(sentences, word2idx, unk_index)

        return {
            "sentences": sentences,
            "embeddings": embeddings,
            "word2idx": word2idx,
            "extra": extra,
            "metadata": metadata,
        }


class ReconstructDataset(object):

    def initialize(self, options, text_path=None, embeddings_path=None, filter_length=0, data_type=None):
        if data_type in ('txt', 'txt_id'):
            reader = PlainTextReader(lowercase=options.lowercase, filter_length=filter_length,
                                     include_label=options.include_label, include_id=data_type == 'txt_id')
        elif data_type == 'wsj_emnlp':
            reader = WSJEMNLPReader(lowercase=options.lowercase, filter_length=filter_length)

        manager = ReaderManager(reader)
        result = manager.run(options, text_path, embeddings_path)

        return result


def make_batch_iterator(options, dset, shuffle=True, include_partial=False, filter_length=0,
                        batch_size=None, length_to_size=None, contextual_target=False, curriculum_start_length=False):
    sentences = dset['sentences']
    word2idx = dset['word2idx']
    extra = dset['extra']
    metadata = dset['metadata']

    n_etypes = metadata.get('n_etypes', None)
    etype2idx = metadata.get('etype2idx', None)

    cuda = options.cuda

    vocab_size = len(word2idx)

    batch_iterator = BatchIterator(
        sentences,
        extra=extra,
        shuffle=shuffle,
        include_partial=include_partial,
        filter_length=filter_length,
        batch_size=batch_size,
        cuda=cuda,
        size=options.hidden_dim,
        word2idx=word2idx,
        options_path=options.elmo_options_path,
        weights_path=options.elmo_weights_path,
        )

    # DIRTY HACK: Makes it easier to print examples later. Should really wrap this within the class.
    batch_iterator.word2idx = word2idx
    batch_iterator.n_etypes = n_etypes
    batch_iterator.etype2idx = etype2idx

    return batch_iterator

