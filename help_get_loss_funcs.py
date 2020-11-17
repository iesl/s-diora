import json

from input_layer import *
from loss_reconstruct import ReconstructFixedVocab
from loss_greedy_reconstruct import GreedyReconstruct


def get_name_to_clz():
    return {
        'reconstruct_softmax_v2': ReconstructFixedVocab,
        'greedy_reconstruct_loss': GreedyReconstruct,
    }


class BuildComponent(object):
    def __init__(self, name, kwargs_dict):
        self.name = name
        self.kwargs_dict = kwargs_dict
        self.logger = get_logger()

    def build(self, context):
        clz = get_name_to_clz()[self.name]
        self.logger.info('building loss component name = {}, class = {}'.format(self.name, clz))
        self.logger.info('with kwargs = {}'.format(self.kwargs_dict))
        return clz.from_kwargs_dict(context, self.kwargs_dict)


def get_default_configs(options, context):
    mlp_input_dim = options.hidden_dim

    return {
        'reconstruct_softmax_v2': dict(input_size=options.input_dim, size=mlp_input_dim),
        'greedy_reconstruct_loss': dict(input_size=options.input_dim, size=mlp_input_dim),
    }


def get_loss_funcs(options, context, config_lst=None):
    assert isinstance(config_lst, (list, tuple)), "There should be a `list` of configs."

    result = []

    for i, config_str in enumerate(config_lst):
        config = json.loads(config_str)

        assert isinstance(config, dict), "Config[{}] with value {} is not type dict.".format(i, config)

        assert len(config.keys()) == 1, "Each config should have 1 key only."

        name = list(config.keys())[0]

        if not config[name].get('enabled', True):
            continue

        # Use default config if it exists.
        kwargs_dict = get_default_configs(options, context).get(name, {})
        # Override defaults with user-defined values.
        for k, v in config[name].items():
            kwargs_dict[k] = v
        # Build and append component.
        result.append(BuildComponent(name, kwargs_dict).build(context))

    return result
