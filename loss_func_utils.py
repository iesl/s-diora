import torch
import torch.nn as nn

from experiment_logger import get_logger


class BaseLossFunc(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        # defaults
        self.enabled = True
        self.unregistered = {}
        self.logger = get_logger()
        self.init_defaults()

        # override
        for k, v in kwargs.items():
            if k in self.unregistered:
                self.unregistered[k] = v
                continue
            setattr(self, k, v)

        self.init_parameters()
        self.reset_parameters()

    @classmethod
    def from_kwargs_dict(cls, context, kwargs_dict):
        kwargs_dict['_cuda'] = context['cuda']
        return cls(**kwargs_dict)

    def init_defaults(self):
        pass

    def init_parameters(self):
        pass

    def reset_parameters(self):
        params = [p for p in self.parameters() if p.requires_grad]
        for i, param in enumerate(params):
            param.data.normal_()

    def forward(self, sentences, neg_samples, diora, info, embed=None):
        # loss = 0
        # ret = {}
        # ret[self.name] = loss
        # return loss, ret
        raise NotImplementedError


def scores_for_cross_entropy(sentences, neg_samples, cell, embeddings, mat):
    batch_size, length = sentences.shape
    k = neg_samples.shape[0]
    emb_pos = embeddings(sentences)
    emb_neg = embeddings(neg_samples.unsqueeze(0))
    cell = cell.view(batch_size, length, 1, -1)
    proj_pos = torch.matmul(emb_pos, torch.t(mat))
    proj_neg = torch.matmul(emb_neg, torch.t(mat))
    xp = torch.einsum('abc,abxc->abx', proj_pos, cell)
    xn = torch.einsum('zec,abxc->abe', proj_neg, cell)
    score = torch.cat([xp, xn], 2)
    return score


def scores_for_tokens(tokens, cell, embeddings, mat):
    assert len(tokens.shape) == 1
    batch_size, length, size = cell.shape
    k = tokens.shape[0]
    emb = embeddings(tokens.unsqueeze(0))
    cell = cell.view(batch_size, length, 1, size)
    proj = torch.matmul(emb, torch.t(mat))
    score = torch.einsum('zec,abxc->abe', proj, cell)
    return score


def cross_entropy(sentences, neg_samples, score, mask_duplicates=True):
    batch_size, length = sentences.shape
    k = neg_samples.shape[0]

    # Ignore duplicates of the ground truth from the negative samples.
    mask = sentences.view(batch_size, length, 1) == neg_samples.view(1, 1, k)
    if mask.shape[0] == 0 or not mask_duplicates:
        mask = None

    def logsumexp(x, dim=-1, eps=1e-8, mask=None):
        if mask is None:
            argval, argmax = torch.max(x, dim=dim, keepdim=True)
            exp = torch.exp(x - argval)
        else:
            # Optionally mask elements.
            x = x * 1 # copy x
            x[:, :, 1:][mask] = x.min().data # do this so that max is not effected
            argval, argmax = torch.max(x, dim=dim, keepdim=True)

            exp = torch.exp(x - argval)
            diff = torch.zeros(x.shape, dtype=torch.float, device=x.device)
            diff[:, :, 1:][mask] = exp[:, :, 1:][mask]
            exp = exp - diff # need to mask exp also

        return argval.squeeze(dim) + torch.log(torch.sum(exp, dim=dim) + eps)

    xent = -score[:, :, 0] + logsumexp(score, dim=-1, mask=mask)

    return xent
