'''
convolution that takes variable-length input.

'''
import torch
import torch.nn as nn
import nn.functional as F

class Conv1(nn.Module):

    def __init__(self, n_in, n_hidden, n_out):
        super(Conv1, self).__init__()
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
        
class Conv2(nn.module):

    def __init__(self, n_n_embed, n_e_embed, n_nodes, n_edges):
        super(Conv2, self).__init__()
        #self.m = init_m()
        self.w_e = nn.Parameter(init_w(n_edges, n_e_embed))
        self.w_n = nn.Parameter(init_w(n_nodes, n_n_embed))
        self.w_e2 = nn.Parameter(init_w(n_edges, n_e_embed))
        self.w_n2 = nn.Parameter(init_w(n_nodes, n_n_embed))
        
        #self.lin_e = nn.Linear(n_e_embed, )
        #pool
        
    def init_m(self, n_nodes, n_edges):
        return torch.ones(n_nodes, n_edges)

    def init_w(self, n1, n2):
        return torch.randn(n1, n2)

    def forward(self, e, n, m):
        e = F.bmm(F.bmm(m, e), self.w_e)
        n = F.bmm(F.bmm(torch.t(m), n), self.w_n)
        #e = F.relu(self.lin_e(e))
        #n = F.relu(self.lin_n(n))
        e = F.bmm(F.bmm(m, e), self.w_e2)
        n = F.bmm(F.bmm(torch.t(m), n), self.w_n2)
        
        return e, n
        
def Conv3(nn.Module):
    def __init__(self, n_hidden, embed_dim):
        super__(Conv3, self).__init__()
        self.w = nn.Parameter(torch.zeros([embed_dim, n_hidden], dtype=torch.float32))
    def forward(self, m, e):        
        e = F.bmm(F.bmm(torch.t(m), e), self.w)
