import collections
import errno
import logging
import os
import time

import numpy as np


LOGGING_NAMESPACE = 'diora'


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def configure_experiment(experiment_path):
    mkdir_p(experiment_path)
    log_file = os.path.join(experiment_path, 'experiment.log')
    configure_logger(log_file)


def configure_logger(log_file):
    """
    Simple logging configuration.
    """

    # Create logger.
    logger = logging.getLogger(LOGGING_NAMESPACE)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    # Create file handler.
    #fh = logging.FileHandler(log_file)
    #fh.setLevel(logging.INFO)
    #fh.setFormatter(formatter)
    #logger.addHandler(fh)

    # Also log to console.
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Create file handler.
    #fh = logging.FileHandler(log_file)
    #fh.setLevel(logging.INFO)
    #fh.setFormatter(formatter)
    #logger.addHandler(fh)

    # HACK: Weird fix that counteracts other libraries (i.e. allennlp) modifying
    # the global logger.
    while len(logger.parent.handlers) > 0:
        logger.parent.handlers.pop()

    return logger


def get_logger():
    return logging.getLogger(LOGGING_NAMESPACE)


class Gauge(object):
    def __init__(self):
        self.d = collections.defaultdict(list)

    def update(self, k, v):
        if isinstance(v, (list, tuple)):
            for vv in v:
                self.update(k, vv)
        else:
            self.d[k].append(v)

    def mean(self, k, retain=False):
        v = np.mean(self.d[k])
        if not retain:
            del self.d[k]
        return v

    def get(self, k, retain=False):
        v = self.d[k]
        if not retain:
            del self.d[k]
        return v

    def clear(self):
        keys = list(self.d.keys())
        for k in keys:
            del self.d[k]


class ExperimentLogger(object):
    def __init__(self):
        super(ExperimentLogger, self).__init__()
        self.logger = get_logger()
        self.g = Gauge()

    def record(self, result):
        batch_size, length = result['batch_size'], result['length']
        self.g.update('length', [length] * batch_size)
        for k, v in result['metrics'].items():
            assert not isinstance(v, (list, tuple)), 'Does not support lists.'
            self.g.update(k, [v] * batch_size)

    def log_batch(self, epoch, step, batch_idx):
        logger = self.logger

        # Length Distribution.
        lengths = self.g.get('length')
        length_counts = collections.Counter(lengths)
        length_counts_str = ' '.join(['{}:{}'.format(k, v) for k, v in sorted(length_counts.items(), key=lambda x: x[1])])
        average_length = np.mean(lengths)

        # Metrics.
        keys = list(self.g.d.keys())
        metrics = {k: self.g.mean(k) for k in keys}
        metric_log_prefix = 'Epoch/Step/Batch={}/{}/{}'.format(epoch, step, batch_idx)
        metric_log_body = ' '.join(['{}={:.3f}'.format(k, v) for k, v in metrics.items()])
        metric_log_str = '{} {}'.format(metric_log_prefix, metric_log_body)

        # Log.
        logger.info(metric_log_str)
        logger.info('Average-Length={}'.format(average_length))
        logger.info('Length-Distribution={}'.format(length_counts_str))

        self.g.clear()

        return metrics

    def log_epoch(self, epoch, step):
        logger = self.logger
        logger.info('Epoch/Step={}/{} (End-Of-Epoch)'.format(epoch, step))

    def log_eval(self, loss, metric):
        logger = self.logger
        logger.info('Eval Loss={} Metric={}.'.format(loss, metric))
