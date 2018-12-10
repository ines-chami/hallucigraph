import torch
from torch.autograd import Variable


def reparameterize(training, mu, logvar):
    if training:
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    else:
        return mu


def normalize_adj(adj):
    row_sum = torch.pow(adj.sum(1), -0.5).flatten()
    adj_norm = adj * row_sum.reshape((1, -1))
    adj_norm *= row_sum.reshape((-1, 1))
    return adj_norm


def sample_graph(adj, temp=0.5, hard=False):
    def sample_gumbel(shape, eps=1e-20):
        """Sample from Gumbel(0, 1)"""
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)

    pos_logits = torch.log(adj + 10e-8)
    neg_logits = torch.log(1 - adj + 10e-8)
    pos_gumbel_noise = Variable(sample_gumbel(pos_logits.size()))
    neg_gumbel_noise = Variable(sample_gumbel(neg_logits.size()))
    y_pos = torch.exp((pos_logits + pos_gumbel_noise) / temp)
    y_neg = torch.exp((neg_logits + neg_gumbel_noise) / temp)
    sample = y_pos / (y_pos + y_neg + 10e-8)
    if hard:
        hard_sample = (sample > 0.5).type(sample.dtype)
        sample = (hard_sample - sample).detach() + sample
    return sample
