
'''
Sequence-to-sequence using convolution
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

#trains latent vectors to capture relations between tokens,
#which the decoder interprets
class Encode(nn.module):
    def __init__(self, n_in, n_hidden):
        super(Encode, self).__init__()
        #self.conv = nn.Conv1D (n_filter, n_kern )
        self.lin1 = nn.Linear(n_in, n_hidden)
        self.lin2 = nn.Linear(n_in, n_hidden)

    #vertices are batches of embeddings of tokens
    def forward(self, vertices):
        #convolve and then sum
        #This needs to be 1D convolution for each sentence,
        #i.e. each row
        v = self.conv( vertices  )
        
        v = F.sum(v, dim=1 )
        #form edge representation , create A
        
        mean = F.relu(self.lin1(v))
        var = F.relu(self.lin2(v))
        return mean, var

class Decode(nn.module):
    def __init__(self, n_in, n_hidden):
        super(Decode, self).__init__()        
        self.lin = nn.Linear(n_in, n_hidden)

    #vertices are batches of embeddings of tokens
    def forward(self, x):
        #convolve and then sum
        #This needs to be 1D convolution for each sentence,
        #i.e. each row
        x = self.relu(self.lin(x))
        
        return x
    
def reparametrize(mean, var):
    #this var is log variance, so *.5 
    noise = torch.randn(torch.exp(var*0.5))    
    return noise.mul(torch.exp(var*0.5)).add(mean)
