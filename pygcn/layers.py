import math

import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout, act, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)
        # if self.bias is not None:
        #     self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, training=self.training)
        support = torch.mm(input, self.weight)
        support = F.dropout(support, self.dropout, training=self.training)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return self.act(output + self.bias)
        else:
            return self.act(output)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class InnerProductDecoder(Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj


class GraphiteDecoder(Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, in_features, out_features, dropout, act=torch.sigmoid):
        super(GraphiteDecoder, self).__init__()
        self.dropout = dropout
        self.act = act
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, z, recon_1, recon_2):
        z = F.dropout(z, self.dropout, training=self.training)
        outputs = torch.mm(z, self.weight)
        outputs = recon_1.dot((torch.transpose(recon_1).dot(outputs) + recon_2.dot(
                torch.transpose(recon_2).dot(outputs))))
        outputs = self.act(outputs)
        return outputs
