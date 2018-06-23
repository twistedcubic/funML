
'''
Models that learn function approximation, starting with polynomials.
Use a basis of monomials, e.g. 1, x, x^2, etc.

'''
import torch
import torch.nn as nn
import nn.functional as F
import torch.optim as optim

class PolyNet(nn.Module):
    def __init__(self, nx):
        super(PolyNet, self).__init__()
        #this is just doing regression, the weights in the transition matrix
        #are the coefficients of the monomials.
        self.lin = nn.Linear(nx, 1)

    #x consists of coefficients of the basis elements in the polynomial
    def forward(self, x):        
        out = self.lin(x)
        return out

    

'''
Model to predict polynomial degree, given values 
 dependent variable y for a given sequence of independent variables.
I.e. learn shape of y on a given sequence of values of x.
First start with monomials in one variable, e.g. x^2, (2x)^4.
'''
class DegreeNet(nn.Module):
    
    def __init__(self, nin, nhidden, nout):
        super(DegreeNet, self).__init__()
        #linear layers to learn the shape of y
        self.lin1 = nn.Linear(nin, nhidden)
        self.lin2 = nn.Linear(nhidden, nout)        
        
    #y values for fixed sequence of x
    def forward(self, input):
        input = self.lin1(input)
        out = F.relu(input)
        out = self.lin2(out)
        out = F.log_softmax(F.relu(out))
        return out
    
#create data given desired sequence length 
def create_data(x, p):   
    return p(x)
    

x = torch.FloatTensor(range(-5,5))/100
y = create_data(x, p)

