'''
Graph convolution that takes variable-length input.

'''
import torch
import torch.nn as nn
import nn.functional as F

class GraphConv1(nn.Module):

    def __init__(self, n_in, n_hidden, n_out):
        super(GraphConv, self).__init__()
        self.lin1 = nn.Linear(n_in, n_hidden)
        self.lin2 = nn.Linear(n_in, n_hidden)
        self.lin = nn.Linear(n_hidden*2, n_out)        

    def forward(self, vertices, edges):
        #train the weights for vertices and edges separately.
        #Take into account both vertices and edgees info, rather than
        #just vertices as in convolving using adjacency matrix.
        v = vertices.sum(dim=1)
        e = edges.sum(dim=1)
        v = F.relu(self.lin1(v))
        e = F.relu(self.lin2(e))
        
        m = F.cat(v, e, dim=1)
        m = F.relu(self.lin(m))
        return m
        
