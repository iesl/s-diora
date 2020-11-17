import torch

from dataset import ReconstructDataset, make_batch_iterator
from dataset import ConsolidateDatasets


def get_train_dataset(options):
    return ReconstructDataset().initialize(options, text_path=options.train_path,
        embeddings_path=options.embeddings_path, filter_length=options.train_filter_length,
        data_type=options.train_data_type)


def get_train_iterator(options, dataset):
    return make_batch_iterator(options, dataset, shuffle=True,
        include_partial=False, filter_length=options.train_filter_length,
        batch_size=options.batch_size)


def get_validation_dataset(options):
    return ReconstructDataset().initialize(options, text_path=options.validation_path,
        embeddings_path=options.embeddings_path, filter_length=options.validation_filter_length,
        data_type=options.validation_data_type)


def get_validation_iterator(options, dataset):
    return make_batch_iterator(options, dataset, shuffle=False,
        include_partial=True, filter_length=options.validation_filter_length,
        batch_size=options.validation_batch_size)


def get_train_and_validation(options):
    train_dataset = get_train_dataset(options)
    validation_dataset = get_validation_dataset(options)

    # Modifies datasets. Unifying word mappings, embeddings, etc.
    ConsolidateDatasets([train_dataset, validation_dataset]).run()

    return train_dataset, validation_dataset
