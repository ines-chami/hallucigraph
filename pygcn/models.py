import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from pygcn.layers import GraphConvolution, InnerProductDecoder


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid, dropout, F.relu)
        self.gc2 = GraphConvolution(nhid, nclass, dropout, lambda x: x)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class Hallucigraph(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(Hallucigraph, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.yygc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class VGAE(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(VGAE, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid, dropout, act=F.relu)
        self.gc2 = GraphConvolution(nhid, nhid, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(nhid, nhid, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1.forward(x, adj)
        return self.gc2.forward(hidden1, adj), self.gc3.forward(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return self.dc.forward(z), mu, logvar


class Graphite(nn.Module):
    def __init__(self, nfeat, nhid, dropout, autoregressive_scalar=0.5):
        super(Graphite, self).__init__()
        self.autoregressive_scalar = autoregressive_scalar
        # encoder
        self.gc1 = GraphConvolution(nfeat, nhid, dropout, act=F.relu)
        self.gc2 = GraphConvolution(nhid, nhid, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(nhid, nhid, dropout, act=lambda x: x)
        # decoder
        self.gd1 = GraphConvolution(nhid, nhid, dropout, act=F.relu)
        self.gd2 = GraphConvolution(nfeat, nhid, dropout, act=F.relu)
        self.gd3 = GraphConvolution(nhid, nhid, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1.forward(x, adj)
        return self.gc2.forward(hidden1, adj), self.gc3.forward(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def _normalize_adj(self, adj):
        row_sum = torch.pow(adj.sum(1), -0.5).flatten()
        adj_norm = adj * row_sum.reshape((1, -1))
        adj_norm *= row_sum.reshape((-1, 1))
        return adj_norm

    def sample_gumbel(self, shape, eps=1e-20):
        """Sample from Gumbel(0, 1)"""
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)

    def sample_graph(self, adj, temp=0.5, hard=False):
        pos_logits = torch.log(adj + 10e-8)
        neg_logits = torch.log(1 - adj + 10e-8)
        pos_gumbel_noise = Variable(self.sample_gumbel(pos_logits.size()))
        neg_gumbel_noise = Variable(self.sample_gumbel(neg_logits.size()))
        y_pos = torch.exp((pos_logits + pos_gumbel_noise) / temp)
        y_neg = torch.exp((neg_logits + neg_gumbel_noise) / temp)
        sample = y_pos / (y_pos + y_neg + 10e-8)
        if hard:
            hard_sample = (sample > 0.5).type(sample.dtype)
            sample = (hard_sample - sample).detach() + sample
        return sample

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z0 = self.reparameterize(mu, logvar)
        # z0 = F.dropout(z0, self.dropout, self.training)
        adj = torch.sigmoid(torch.mm(z0, z0.t()))
        adj = self.sample_graph(adj)
        adj_norm = self._normalize_adj(adj)
        z1 = self.gd1.forward(z0, adj_norm)
        z2 = self.gd2.forward(x, adj_norm)
        z3 = self.gd3.forward(z1 + z2, adj_norm)
        z4 = (1 - self.autoregressive_scalar) * z0 + self.autoregressive_scalar * z3
        return self.dc.forward(z4), mu, logvar
