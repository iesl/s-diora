import json

import torch
import torch.nn as nn
import torch.optim as optim

from allennlp.modules.elmo import Elmo
from allennlp.modules.elmo import Elmo, batch_to_ids

from embeddings import initialize_word2idx, BOS_TOKEN, EOS_TOKEN
from experiment_logger import get_logger


def get_embed_and_project(options, embeddings, input_dim, size, word2idx=None, contextual=False):
    if options.projection == 'word2vec':
        projection_layer, embedding_layer = word2vec_projection(options, embeddings, input_dim, size, word2idx)
    return projection_layer, embedding_layer


def word2vec_projection(options, embeddings, input_dim, size, word2idx=None):
    embedding_layer = nn.Embedding.from_pretrained(torch.from_numpy(embeddings), freeze=True)
    elmo = None
    projection_layer = EmbedAndProject(embedding_layer, input_size=input_dim, size=size)
    return projection_layer, embedding_layer


class EmbedAndProject(nn.Module):
    def __init__(self, embeddings, input_size, size):
        super().__init__()
        self.input_size = input_size
        self.size = size
        self.embeddings = embeddings
        self.mat = nn.Parameter(torch.FloatTensor(size, input_size))
        self.reset_parameters()

    def reset_parameters(self):
        params = [p for p in self.parameters() if p.requires_grad]
        for i, param in enumerate(params):
            param.data.normal_()

    def embed(self, x):
        """ Always context-insensitive embedding.
        """
        return self.embeddings(x)

    def project(self, x):
        return torch.matmul(x, self.mat.t())

    def forward(self, x, info=None):
        batch_size, length = x.shape

        embed = self.embed(x)

        out = embed

        return self.project(out)
