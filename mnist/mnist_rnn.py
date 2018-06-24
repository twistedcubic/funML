
'''
Uses Elman RNN to classify mnist data. Initialize the RNN weights

'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

embed_dim = 128
#dimension of feature space, i.e. number of different tokens.
feature_dim = 4000
#number of possible outputs
output_dim = 10
hidden_dim = 32
#number of filters
conv_channels_in = 8
conv_channels_out = 4
conv_kern_sz = 3
pool_kern_sz = 5
conv_lin_sz = embed_dim * conv_channels_out

class MnistRnn(nn.Module):
    def __init__(self):
        super(MnistRnn, self).__init__()
        #first use embedding layer to encode tokens
        self.embed = nn.Embedding(feature_dim, embed_dim)

        self.rnn = nn.RNN(input_size=embed_dim, hidden_size=hidden_dim, nonlinearity='relu',
                          batch_first=True)
        
        self.hidden = init_hidden(hidden_dim)
        self.lin = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    #Initialize the weights and biases to identity and zero, resp.
    #ref. https://arxiv.org/abs/1504.00941        
    #batch_first in the RNN is set to True
    def init_weight_bias(self, ninput, nhidden, nlayer=1):
        #shape: weight_ih_l[k] is the weights of the k_th layer.
        #The shape is (nhidden*ninput) for k = 0, and (nhidden*nhidden) otherwise.
        self.rnn.weight_ih_l[0].data.copy_(torch.eye(ninput, nhidden))
        self.rnn.bias_ih_l[0].data.copy_(torch.zeros(nhidden))
        
        for i in range(1, nlayer):
            self.rnn.weight_ih_l[i].data.copy_(torch.eye(nhidden, nhidden))
            self.rnn.bias_ih_l[i].data.copy_(torch.zeros(nhidden))            
        
    #Args: x is batched sequences of integers, each sequence is a sentence,
    #each integer represents a token.
    def forward(self, x):
        x = self.embed(x)
        x, hidden = self.rnn(x, self.hidden)
        x = self.lin(x)
        return self.softmax(x)

    #batch_first is set to True
    def init_hidden(self, nbatch, hidden_dim, seq_len=1):
        return torch.zeros(nbatch, seq_len, hidden_dim)

model = MnistRnn()
opt = optim.SGD(model.parameters(), lr = 0.005, momentum = 0.9)
criterion = nn.NLLLoss()

'''
Args: x consists of sequences of integers, y consists of one-hot vectors representing
labels (i.e. 1 at label index, 0 elsewhere).
'''
def train(x, y):
    opt.zero_grad()
    #initialize weights to apply to input, as well as weights applied to hidden state
    model.init_weight_bias(feature_dim, hidden_dim)
    predicted = model(x)
    loss = criterion(predicted, y)
    loss.backward()
    opt.step()
