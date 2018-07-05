

'''
Variational autoencoder. The goal is to find the parameters that 
maximize conditional probability of the data. Specifically,
log(\int p(x|z) dz), which is bounded below by E[log(p(x|z))] + KL(q(z|x)||p(z)).
Encoder encodes input into latent vectors, and decoder predicts based on these
latent vectors.
'''

import torch
import torch.nn as nn
import torch.functional as F

class VaeEnc(nn.module):
    def __init__(self, nin, nhidden):
        super(VaeEnc, self).__init__()        
        self.lin1 = nn.Linear(nin, nhidden)
        self.lin2 = nn.Linear(nin, nhidden)
        
    def forward(self, x):
        mean = F.sigmoid(self.lin1(x))
        var = F.sigmoid(self.lin2(x))
        return mean, var
    
class VaeDec(nn.module):
    def __init__(self, nhidden, nout):
        super(VaeDec, self).__init__()
        self.lin = nn.Linear(nhidden, nout)

    def forward(self, x):
        x = F.sigmoid(self.lin(x))
        return x

#Reparametrization trick.
#reparametrize based on data mean and variance.
def reparametrize(mean, var):
    noise = torch.randn(mean.size(0), mean.size(1))
    return noise.mul(torch.exp(0.5*var)).add(mean)
    
class Vae(nn.module):
    def __init__(self, nin, nhidden, nout):
        super(Vae, self).__init__()
        self.enc = VaeEnc(nin, nhidden)
        self.dec = VaeDec(nhidden, nout)

    def forward(self, x):
        mean, var = self.enc(x)
        x = reparametrize(mean, var)
        x = self.dec(x)
        return x, mean, var
