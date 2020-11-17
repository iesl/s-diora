from dataloader import FixedLengthBatchSampler, SimpleDataset
from experiment_logger import get_logger

from allennlp.modules.elmo import Elmo, batch_to_ids

import torch
import numpy as np
import json
from tqdm import tqdm


def get_config(config, **kwargs):
    for k, v in kwargs.items():
        if k in config:
            config[k] = v
        else:
            pass
            # raise ValueError('Invalid keyword "{}"'.format(k))
    return config


def get_default_config():

    default_config = dict(
        batch_size=16,
        forever=False,
        drop_last=False,
        sort_by_length=True,
        shuffle=True,
        random_seed=None,
        filter_length=None,
        workers=10,
        include_partial=False,
        cuda=False,
        options_path=None,
        weights_path=None,
        word2idx=None,
        size=None,
    )

    return default_config


def indexify(raw_sentences, word2idx):
    def func():
        for s in raw_sentences:
            yield [word2idx[w] for w in s]
    return torch.LongTensor(list(func()))


class BatchIterator(object):

    def __init__(self, sentences, extra={}, **kwargs):
        self.sentences = sentences
        self.config = config = get_config(get_default_config(), **kwargs)
        self.extra = extra
        self.loader = None
        self.cuda = config.get('cuda')
        self.logger = get_logger()

    def id2raw(self, batch):
        word2idx = self.config['word2idx']
        idx2word = {v: k for k, v in word2idx.items()}
        def convert(s):
            return [idx2word[w] for w in s]
        xs = batch.tolist()
        ws = [convert(s) for s in xs]
        return ws

    def get_dataset_size(self):
        return len(self.sentences)

    def get_dataset_minlen(self):
        return min(map(len, self.sentences))

    def get_dataset_maxlen(self):
        return max(map(len, self.sentences))

    def get_dataset_stats(self):
        return 'size={} minlen={} maxlen={}'.format(
            self.get_dataset_size(), self.get_dataset_minlen(), self.get_dataset_maxlen()
        )

    def get_vectors(self, *args, **kwargs):
        return self.vector_cache.get_vectors(*args, **kwargs)

    def save_vectors(self, *args, **kwargs):
        self.vector_cache.save_vectors(*args, **kwargs)

    def get_iterator(self, **kwargs):
        config = get_config(self.config.copy(), **kwargs)

        random_seed = config.get('random_seed')
        batch_size = config.get('batch_size')
        filter_length = config.get('filter_length')
        include_partial = config.get('include_partial')
        cuda = config.get('cuda')
        workers = config.get('workers')

        epoch = kwargs.get('epoch', 0)
        length_limit = 1000

        def collate_fn(batch):
            index, sents = zip(*batch)
            sents = torch.from_numpy(np.array(sents)).long()

            batch_map = {}
            batch_map['index'] = index
            batch_map['sents'] = sents

            for k, v in self.extra.items():
                batch_map[k] = [v[idx] for idx in index]

            return batch_map

        if self.loader is None:
            rng = np.random.RandomState(seed=random_seed)
            dataset = SimpleDataset(self.sentences)
            sampler = FixedLengthBatchSampler(dataset, batch_size=batch_size, rng=rng,
                include_partial=include_partial)
            # TODO: workers
            loader = torch.utils.data.DataLoader(dataset, shuffle=(sampler is None), num_workers=2, batch_sampler=sampler, collate_fn=collate_fn)
            self.loader = loader

        def myiterator():

            for i, batch in enumerate(self.loader):
                index = batch['index']
                sentences = batch['sents']

                batch_size, length = sentences.shape
                if length > length_limit:
                    continue

                if cuda:
                    sentences = sentences.cuda()

                batch_map = {}
                batch_map['sentences'] = sentences
                batch_map['batch_size'] = batch_size
                batch_map['length'] = length

                for k, v in self.extra.items():
                    batch_map[k] = batch[k]

                yield batch_map

        return myiterator()

