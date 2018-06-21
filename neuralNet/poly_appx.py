
'''
Models that learn function approximation, starting with polynomials.
Use a basis of monomials.
'''
import torch
import torch.nn as nn
import nn.functional as F
import torch.optim as optim

class PolyNet(nn.Module):
    def __init__(self, nx, hidden):
        super(PolyNet, self).__init__()
        self.lin1 = nn.Linear(nx, hidden)
        self.lin2 = nn.Linear(hidden, 1)

    #x consists of coefficients of the basis elements in the polynomial
    def forward(self, x):
        x = F.relu(self.lin1(x))
        out = self.lin2(x)
        return out
