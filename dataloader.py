import os
from collections import OrderedDict

import torch
from torch.utils.data import Sampler

import numpy as np

from experiment_logger import get_logger


class FixedLengthBatchSampler(Sampler):

    def __init__(self, data_source, batch_size, include_partial=False, rng=None):
        self.data_source = data_source
        self.active = False
        if rng is None:
            rng = np.random.RandomState(seed=11)
        self.rng = rng
        self.batch_size = batch_size
        self.include_partial = include_partial
        self._batch_size_cache = { 0: self.batch_size }
        self.logger = get_logger()

    def reset(self):
        # Record the lengths of each example.
        length_map = OrderedDict()
        for i in range(len(self.data_source)):
            x = self.data_source.dataset[i]
            length_map.setdefault(len(x), []).append(i)

        # Shuffle the order.
        for length in length_map.keys():
            self.rng.shuffle(length_map[length])

        # Initialize state.
        state = {}
        for length, arr in length_map.items():
            batch_size = self.batch_size
            nbatches = len(arr) // batch_size
            surplus = nbatches * batch_size < len(arr)
            state[length] = dict(nbatches=nbatches, surplus=surplus, position=-1)

        # Batch order, in terms of length.
        order = []
        for length, v in state.items():
            order += [length] * v['nbatches']

        ## Optionally, add partial batches.
        if self.include_partial:
            for length, v in state.items():
                if v['surplus']:
                    order += [length]

        self.logger.info('# of batches = {}'.format(len(order)))

        self.rng.shuffle(order)

        self.length_map = length_map
        self.state = state
        self.order = order
        self.index = -1

    def get_next_batch(self, length):
        batch_size = self.batch_size
        position = self.state[length]['position'] + 1
        start = position * batch_size
        batch_index = self.length_map[length][start:start+batch_size]

        self.state[length]['position'] = position

        return batch_index

    def __iter__(self):
        self.reset()

        for _ in range(len(self)):
            index = self.index + 1
            length = self.order[index]
            self.index = index
            yield self.get_next_batch(length)

    def __len__(self):
        return len(self.order)


class SimpleDataset(torch.utils.data.Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        item = self.dataset[index]
        return index, item

    def __len__(self):
        return len(self.dataset)

