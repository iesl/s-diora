import collections
import copy
import json
import os
import sys
import traceback
import types

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from allennlp.nn.util import add_positional_features

from experiment_logger import get_logger
from embeddings import initialize_word2idx
from help_get_net_components import check_params, nested_getattr


def default_getattr(o, k, default=None):
    if not hasattr(o, k):
        return default
    return getattr(o, k)


def nested_getattr(o, k):
    k_lst = k.split('.')
    for i in range(len(k_lst)):
        o = getattr(o, k_lst[i])
    return o


def nested_hasattr(o, k):
    try:
        _ = nested_getattr(o, k)
        return True
    except:
        return False


def nested_setattr(o, k, v):
    k_lst = k.split('.')
    if len(k_lst) > 1:
        new_k = '.'.join(k_lst[:-1])
        o = nested_getattr(o, new_k)
    setattr(o, k_lst[-1], v)


class Net(nn.Module):
    def __init__(self, embed=None, diora=None, loss_funcs=[], size=None):
        super(Net, self).__init__()

        self.cache = {}

        self.embed = embed
        self.diora = diora

        self.size = size

        self.loss_func_names = []
        self.add_loss_funcs(loss_funcs)

    def add_loss_funcs(self, loss_funcs):
        for m in loss_funcs:
            # Assign.
            setattr(self, m.name, m)
            self.loss_func_names.append(m.name)

    @property
    def is_cuda(self):
        return self.diora.is_cuda

    def compute_loss(self, batch, diora, info, embed, name_filter=None):
        device = torch.cuda.current_device() if self.is_cuda else None
        info['net'] = self
        ret, loss = {}, []

        func_name_lst = self.loss_func_names
        if name_filter is not None:
            func_name_lst = [x for x in func_name_lst if x in name_filter]

        # Loss
        for func_name in func_name_lst:
            func = getattr(self, func_name)
            if default_getattr(func, 'skip', False):
                continue
            subloss, desc = func(batch, diora, info, embed)
            loss.append(subloss.view(1, 1))
            for k, v in desc.items():
                ret[k] = v

        if len(loss) > 0:
            loss = torch.cat(loss, 1)
        else:
            loss = torch.full((1, 1), 1, dtype=torch.float, device=device)

        return ret, loss

    def forward(self, batch, compute_loss=True, info={}):
        device = torch.cuda.current_device() if self.is_cuda else None

        # Reset cache.
        for k, v in list(self.cache.items()):
            del self.cache[k]

        # Embed
        embed = self.embed(batch, info)

        # Run DIORA
        diora = self.diora
        chart = diora(embed, info=info)

        # Compute Loss
        if isinstance(compute_loss, (tuple, list, str)):
            if isinstance(compute_loss, str):
                compute_loss = [compute_loss]
            ret, loss = self.compute_loss(batch, diora=diora, info=info, embed=embed, name_filter=compute_loss)
        elif compute_loss:
            ret, loss = self.compute_loss(batch, diora=diora, info=info, embed=embed)
        else:
            ret, loss = {}, torch.full((1, 1), 1, dtype=torch.float, device=device)

        # Results
        ret['chart'] = chart
        ret['total_loss'] = loss

        return ret


class Trainer(object):
    def __init__(self, net=None, cuda=None, word2idx=None, mlp_reg=0, clip_threshold=5.0):

        super(Trainer, self).__init__()
        self.net = net
        self.optimizer = None
        self.optimizer_cls = None
        self.optimizer_kwargs = None
        self.cuda = cuda
        self.word2idx = word2idx
        self._embedding_keys = None
        self.mlp_reg = mlp_reg
        self.clip_threshold = clip_threshold

        self.logger = get_logger()

    def save_init_parameters(self, net):
        d = {}
        for n, p in net.named_parameters():
            if not p.requires_grad:
                continue
            phat = p.clone().detach()
            phat.requires_grad = False
            d[n] = phat
        self.init_parameters = d

    @staticmethod
    def get_single_net(net):
        if isinstance(net, torch.nn.parallel.DistributedDataParallel):
            return net.module
        return net

    def embedding_keys(self, net):
        if self._embedding_keys is not None:
            # Cache these keys because they will be deleted.
            return self._embedding_keys
        suffix = '.weight'
        keys = [k[:len(k)-len(suffix)] for k in net.state_dict().keys() if 'embeddings.weight' in k]
        self._embedding_keys = keys
        return keys

    def init_optimizer(self, options):
        if options.opt == 'adam':
            optimizer_cls = optim.Adam
            optimizer_kwargs = dict(lr=options.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=options.weight_decay)
        elif options.opt == 'sgd':
            optimizer_cls = optim.SGD
            optimizer_kwargs = dict(lr=options.lr, weight_decay=options.weight_decay)

        if optimizer_cls is None:
            optimizer_cls = self.optimizer_cls
        if optimizer_kwargs is None:
            optimizer_kwargs = self.optimizer_kwargs

        trainable_params = [p for (name, p) in self.net.named_parameters() if p.requires_grad]
        self.optimizer = optimizer_cls(trainable_params, **optimizer_kwargs)

    def save_model(self, model_file):
        state_dict = self.net.state_dict()

        todelete = []

        for k in state_dict.keys():
            if 'embeddings' in k:
                todelete.append(k)

        for k in todelete:
            del state_dict[k]

        torch.save({
            'state_dict': state_dict,
            'word2idx': self.word2idx,
        }, model_file)

    @staticmethod
    def load_model(net, model_file):
        save_dict = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict_toload = save_dict['state_dict']
        state_dict_net = Trainer.get_single_net(net).state_dict()

        print('[load] state dict keys: {}'.format(state_dict_toload.keys()))
        print('[net] state dict keys: {}'.format(state_dict_net.keys()))

        todelete = []

        # Rename some layers.
        keys = list(state_dict_toload.keys())
        for k in keys:
            if k.startswith('module'):
                newk = k[len('module.'):]
                state_dict_toload[newk] = state_dict_toload[k]
                del state_dict_toload[k]

        # Deprecated MLP.
        keys = list(state_dict_toload.keys())
        for k in keys:
            newk = None
            if k == 'diora.inside_compose_func.W_0':
                newk = 'diora.inside_compose_func.W'
            if k == 'diora.outside_compose_func.W_0':
                newk = 'diora.outside_compose_func.W'
            if k == 'diora.atten_func.mat':
                newk = 'diora.atten_func.0.mat'
            if newk is not None:
                state_dict_toload[newk] = state_dict_toload[k]
                del state_dict_toload[k]

        state_dict_toload_copy = state_dict_toload.copy()

        # Remove extra keys.
        keys = list(state_dict_toload.keys())
        for k in keys:
            # print('state_dict[load]', k, k in state_dict_net)
            if k not in state_dict_net:
                print('deleting (missing from state_dict) {}'.format(k))
                del state_dict_toload[k]

        for k in todelete:
            print('deleting (during load) {}'.format(k))
            del state_dict_toload[k]

        # Hack to support embeddings.
        for k in state_dict_net.keys():
            if 'embeddings' in k:
                print('restoring {}'.format(k))
                state_dict_toload[k] = state_dict_net[k]

        # Add any missing modules.
        for k in state_dict_net.keys():
            if k not in state_dict_toload:
                print('add missing {}'.format(k))
                state_dict_toload[k] = state_dict_net[k]

        Trainer.get_single_net(net).load_state_dict(state_dict_toload)

    def run_net(self, batch_map, compute_loss=True, info={}):
        batch = batch_map['sentences']
        for k, v in self.prepare_info(batch_map).items():
            info[k] = v

        if self.cuda:
            batch = batch.cuda()

        net = self.net

        out = net(batch, compute_loss=compute_loss, info=info)

        return out

    def gradient_update(self, loss):
        net = self.net

        self.optimizer.zero_grad()

        loss.backward()

        trainable_params = [p for p in net.parameters() if p.requires_grad]
        clip_threshold = np.inf if self.clip_threshold == 0 else self.clip_threshold
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, clip_threshold)

        self.optimizer.step()

    def prepare_result(self, batch_map, model_output):
        result = {}
        result['batch_size'] = batch_map['batch_size']
        result['length'] = batch_map['length']
        result['inside_root'] = model_output.get('inside_root', None)
        result['scores'] = model_output.get('scores', None)
        result['samples'] = model_output.get('samples', None)

        metrics = collections.OrderedDict()
        for k, v in model_output.items():
            if 'loss' in k:
                metrics[k] = v.mean(dim=0).sum().item()
            elif 'metric' in k:
                metrics[k] = v
            elif 'acc' in k:
                metrics[k] = v.mean(dim=0).item()
        result['metrics'] = metrics

        return result

    def prepare_info(self, batch_map):
        info = {}
        if 'raw_parse' in batch_map:
            info['raw_parse'] = batch_map['raw_parse']
        if 'constituency_tags' in batch_map:
            info['constituency_tags'] = batch_map['constituency_tags']
        if 'entity_labels' in batch_map:
            info['entity_labels'] = batch_map['entity_labels']
        if 'pos_tags' in batch_map:
            info['pos_tags'] = batch_map['pos_tags']
        if 'binary_tree' in batch_map:
            info['binary_tree'] = batch_map['binary_tree']
        if 'spans' in batch_map:
            info['spans'] = batch_map['spans']
        if 'span_siblings' in batch_map:
            info['span_siblings'] = batch_map['span_siblings']
        if 'vector_target' in batch_map:
            info['vector_target'] = batch_map['vector_target']
        if 'sentence_idx' in batch_map:
            info['sentence_idx'] = batch_map['sentence_idx']
        if 'sibling_idx' in batch_map:
            info['sibling_idx'] = batch_map['sibling_idx']
        if 'sibling' in batch_map:
            info['sibling'] = batch_map['sibling']
        if 'labels' in batch_map:
            info['labels'] = batch_map['labels']
        if 'example_ids' in batch_map:
            info['example_ids'] = batch_map['example_ids']
        return info

    def cleanup(self, batch_map=None):
        if batch_map is not None:
            for k, v in list(batch_map.items()):
                del batch_map[k]
        self.net.diora.reset()

    def single_step(self, batch_map, train=True, compute_loss=True, info={}):
        if train:
            self.net.train()
        else:
            self.net.eval()

        with torch.set_grad_enabled(train):
            model_output = self.run_net(batch_map, compute_loss=compute_loss, info=info)

        total_loss = model_output['total_loss'].mean(dim=0).sum()

        if train:
            self.gradient_update(total_loss)

        result = self.prepare_result(batch_map, model_output)

        return result

    def step(self, batch_map, train=True, compute_loss=True, info={}):
        return self.single_step(batch_map=batch_map, train=train, compute_loss=compute_loss, info=info)

    def end_of_epoch(self, best_dict):
        pass


def build_net(options, context=dict(), net_components=dict()):

    logger = get_logger()

    logger.info('build net')

    # Components.
    projection_layer = net_components['projection_layer']
    diora = net_components['diora']
    loss_funcs = net_components['loss_funcs']

    # Context.
    cuda = context['cuda']
    lr = options.lr
    word2idx = context['word2idx']

    # Net
    net = Net(projection_layer, diora, loss_funcs=loss_funcs, size=options.hidden_dim)

    # Load model.
    if options.load_model_path is not None:
        logger.info('Loading model: {}'.format(options.load_model_path))
        Trainer.load_model(net, options.load_model_path)

    # CUDA-support
    if cuda:
        net.cuda()
        diora.cuda()

    # Trainer
    trainer_kwargs = {}
    trainer_kwargs['net'] = net
    trainer_kwargs['cuda'] = cuda
    trainer_kwargs['mlp_reg'] = options.mlp_reg
    trainer_kwargs['clip_threshold'] = options.clip_threshold
    trainer_kwargs['word2idx'] = word2idx
    trainer = Trainer(**trainer_kwargs)
    trainer.experiment_name = options.experiment_name

    trainer.init_optimizer(options)

    logger.info('Architecture =\n {}'.format(str(trainer.net)))

    return trainer

