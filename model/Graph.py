# coding: utf-8
# @Author    :陈梦淇
# @time      :2024/3/19
import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2 * in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, h0, lamda, alpha, l):
        # theta = math.log(lamda/l+1)
        theta = min(1, math.log(lamda / l + 1))
        # hi = torch.spmm(adj, input)
        hi = torch.matmul(adj, input)
        if self.variant:
            support = torch.cat([hi, h0], 1)
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support
        # output = theta * torch.mm(support, self.weight) + (1 - theta) * r
        output = theta * torch.matmul(support, self.weight) + (1 - theta) * r
        if self.residual:  # speed up convergence of the training process
            output = output + input
        return output


class deepGCN(nn.Module):
    def __init__(self, nlayers, nfeat, nhidden, nclass, dropout, lamda, alpha, variant):
        super(deepGCN, self).__init__()
        self.hidden_size = nhidden
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden, residual=True, variant=variant))
        self.fcs = nn.ModuleList()
        self.layerNorm = nn.LayerNorm(nfeat)
       
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(256, nclass))
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.lamda = lamda
        self.alpha = alpha
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, adj):
        try:
            _layers = []
            x = F.dropout(x, self.dropout, training=self.training)
            layer_inner = self.act_fn(self.fcs[0](x))
            _layers.append(layer_inner)
            for i, con in enumerate(self.convs):
                layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
                layer_inner = self.act_fn(con(layer_inner, adj, _layers[0], self.lamda, self.alpha, i + 1))
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            # mid_inner = layer_inner  # graph embedding
            layer_inner = self.fcs[-1](layer_inner)
            return layer_inner
        except Exception as e:
            print(1)

    

    def predict(self, x, adj):
        return self.sigmoid(self.forward(x, adj))


if __name__ == "__main__":
    pass
