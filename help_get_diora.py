import json

from diora import DioraMLP
from hard_diora import DioraMLPWithTopk
from experiment_logger import get_logger


def get_diora(options, context, config=None):
    if config is None:
        return legacy_get_diora(options, context)

    config = json.loads(config)

    assert isinstance(config, dict), "Config with value {} is not type dict.".format(i, config)

    assert len(config.keys()) == 1, "Config should have 1 key only."

    name = list(config.keys())[0]

    # Use default config if it exists.
    kwargs_dict = get_default_configs(options, context).get(name, {})
    # Override defaults with user-defined values.
    for k, v in config[name].items():
        kwargs_dict[k] = v
    # Build and return.
    return BuildComponent(name, kwargs_dict).build(context)


class BuildComponent(object):
    def __init__(self, name, kwargs_dict):
        self.name = name
        self.kwargs_dict = kwargs_dict
        self.logger = get_logger()

    def build(self, context):
        kwargs_dict = self.kwargs_dict.copy()
        kwargs_dict['projection_layer'] = None
        clz = name_to_clz(context)[self.name]
        self.logger.info('building diora name = {}, class = {}'.format(self.name, clz))
        self.logger.info('and kwargs = {}'.format(json.dumps(kwargs_dict)))
        return clz.from_kwargs_dict(context, self.kwargs_dict)


def name_to_clz(context):
    return {
        'mlp': DioraMLP,
        'topk-mlp': DioraMLPWithTopk,
    }


def get_default_configs(options, context):
    #
    size = 400
    normalize = 'unit'
    n_layers = 2

    res = {}
    res['mlp'] = dict(size=size, outside=True, normalize=normalize,  n_layers=n_layers)
    res['topk-mlp'] = dict(size=size, outside=True, normalize=normalize, n_layers=n_layers)
    return res
