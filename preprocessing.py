from collections import Counter, OrderedDict
import math
import random

import numpy as np
import torch
from tqdm import tqdm

from embeddings import initialize_word2idx, BOS_TOKEN, EOS_TOKEN


DEFAULT_UNK_INDEX = 1


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_text_vocab(sentences):
    word2idx = initialize_word2idx()
    for s in sentences:
        for w in s:
            if w not in word2idx:
                word2idx[w] = len(word2idx)
    return word2idx

def build_entity_vocab(entity_labels, eid2idx=None):
    def func_etype():
        for lst in entity_labels:
            for el in lst:
                yield el[-1]
    etype_lst = list(sorted(set(x for x in func_etype())))
    etype2idx = {e: i for i, e in enumerate(etype_lst)}
    return etype2idx

def indexify(sentences, word2idx, unk_index=None):
    def fn(s):
        for w in s:
            if w not in word2idx and unk_index is None:
                raise ValueError
            yield word2idx.get(w, unk_index)
    return [list(fn(s)) for s in tqdm(sentences, desc='indexify')]

def indexify_etype(entity_labels, etype2idx):
    def func_index():
        for lst in entity_labels:
            yield [(*el[:-1], etype2idx[el[-1]]) for el in lst if el[-1] in etype2idx]
    return list(func_index())

def batchify(examples, batch_size):
    sorted_examples = list(sorted(examples, key=lambda x: len(x)))
    num_batches = int(math.ceil(len(examples) / batch_size))
    batches = []

    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        batch = sorted_examples[start:end]
        batches.append(pad(batch))

    return batches


def pad(examples, padding_token=0):
    def convert2numpy(batch):
        # Note that this is tranposed to have dimensions (batch_size, sentence_length).
        return np.array(batch, dtype=np.int32).T

    maxlength = np.max([len(x) for x in examples])
    batch = []

    for x in examples:
        diff = maxlength - len(x)
        padded = [0] * diff + x
        batch.append(padded)

    return convert2numpy(batch)


def batch_iterator(dataset, batch_size, seed=None, drop_last=False):
    if seed is not None:
        set_random_seed(seed)

    nexamples = len(dataset)
    nbatches = math.ceil(nexamples/batch_size)
    index = random.sample(range(nexamples), nexamples)

    for i in range(nbatches):
        start = i * batch_size
        end = start + batch_size
        if end > nexamples and drop_last:
            break

        batch = [dataset[i] for i in index[start:end]]
        yield batch


def prepare_batch(batch):
    return batch


def synthesize_training_data(nexamples, vocab_size, min_length=10, max_length=30, seed=None):
    if seed is not None:
        set_random_seed(seed)

    dataset = []

    for i in range(nexamples):
        length = np.random.randint(min_length, max_length)
        example = np.random.randint(0, vocab_size, size=length).tolist()
        dataset.append(example)

    return dataset
