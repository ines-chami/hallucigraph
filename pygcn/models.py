import torch
import torch.nn as nn
import torch.nn.functional as F

from pygcn.layers import GraphConvolution, InnerProductDecoder
from pygcn.utils.model_utils import sample_graph, normalize_adj, reparameterize


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid, dropout, F.relu)
        self.gc2 = GraphConvolution(nhid, nclass, dropout, lambda x: x)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class Hallucigraph(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, autoregressive_scalar=0.5, temp=0.25):
        super(Hallucigraph, self).__init__()
        self.dropout = dropout
        self.autoregressive_scalar = autoregressive_scalar
        self.temp = temp
        self.gc1 = GraphConvolution(nfeat, nhid, dropout=0, act=F.relu)
        self.gc2 = GraphConvolution(nhid, nhid, dropout=0, act=lambda x: x)
        self.gc3 = GraphConvolution(nhid, nclass, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)
        self.gd1 = GraphConvolution(nhid, nhid, dropout, act=F.relu)
        self.gd2 = GraphConvolution(nfeat, nhid, dropout, act=F.relu)
        self.gd3 = GraphConvolution(nhid, nhid, dropout, act=lambda x: x)
        self.gd4 = GraphConvolution(nhid, nclass, dropout, act=lambda x: x)

    def encode(self, x, adj):
        return self.gc2(self.gc1(x, adj), adj)

    def _forward(self, x, adj):
        z0 = self.gc1(x, adj)
        # mu, logvar = self.encode(x, adj)
        # z0 = reparameterize(self.training, mu, logvar)
        preds0 = self.gc3(z0, adj)
        adj_rec = torch.sigmoid(torch.mm(z0, z0.t()))
        adj_rec = sample_graph(adj_rec)
        adj_norm = normalize_adj(adj_rec)
        z1 = self.gd1(z0, adj_norm)
        z2 = self.gd2(x, adj_norm)
        preds1 = self.gd4(z1 + z2, adj_norm)
        preds = (1 - self.autoregressive_scalar) * preds0 + self.autoregressive_scalar * preds1
        return F.log_softmax(preds, dim=1)  # , mu, logvar

    def forward(self, x, adj):
        z0 = self.encode(x, adj)
        adj_scores = torch.mm(z0, z0.t())
        adj_rec = sample_graph(torch.sigmoid(adj_scores), temp=self.temp, hard=False)
        adj_norm = normalize_adj(adj_rec)
        z1 = self.gd1.forward(z0, adj_norm)
        z2 = self.gd2.forward(x, adj_norm)
        z3 = self.gd3(z1 + z2, adj_norm)
        z = (1 - self.autoregressive_scalar) * z0 + self.autoregressive_scalar * z3
        preds = self.gc3(z, adj)
        return adj_scores, F.log_softmax(preds, dim=1)


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

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = reparameterize(self.training, mu, logvar)
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

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z0 = reparameterize(self.training, mu, logvar)
        # z0 = F.dropout(z0, self.dropout, self.training)
        adj = torch.sigmoid(torch.mm(z0, z0.t()))
        adj = sample_graph(adj)
        adj_norm = normalize_adj(adj)
        z1 = self.gd1.forward(z0, adj_norm)
        z2 = self.gd2.forward(x, adj_norm)
        z3 = self.gd3.forward(z1 + z2, adj_norm)
        z4 = (1 - self.autoregressive_scalar) * z0 + self.autoregressive_scalar * z3
        return self.dc.forward(z4), mu, logvar
