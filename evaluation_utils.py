from experiment_logger import get_logger


class BaseEvalFunc(object):
    def __init__(self, **kwargs):
        super().__init__()

        # defaults
        self.enabled = True
        self.is_initialized = True
        self.write = False
        self.logger = get_logger()
        self.init_defaults()

        # override
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def from_kwargs_dict(cls, context, kwargs_dict):
        kwargs_dict['dataset'] = context['dataset']
        kwargs_dict['batch_iterator'] = context['batch_iterator']
        return cls(**kwargs_dict)

    def init_defaults(self):
        pass

    def compare(self, prev_best, results):
        key = None
        value = None
        is_best = False
        return [(key, value, is_best)]
