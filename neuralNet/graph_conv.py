

'''
Graph convolution

'''
import torch
import torch.nn as nn
import nn.functional as F

class GraphConv0(nn.Module):

    def __init__(self, w_row, w_col):
        super(GraphConv0, self).__init__()
        #weights for features*adj matrices
        self.w = torch.randn(w_row, w_col)
                
    '''
    Input consists of feature matrix on all nodes, and 
    adjacency matrix of the graph
    '''
    def forward(self, features, adj):
        #convolve features with nodes one-edge away
        m = nn.bmm(adj, features)
        m = nn.bmm(m, self.w)
        
        m = F.tanh(m)
        return m

'''
This scheme creates a new graph each time, rather than training the same
overall graph. This happens by concatenating the input vectors describing
the connecting edges and neighboring vertices, and feeding them into various
linear layers.
'''
                
