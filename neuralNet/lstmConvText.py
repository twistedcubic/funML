
'''
Net that uses recurrent (LSTM) and convolution layers to
classify text, input as tokens.
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

class RecConv(nn.Module):
    def __init__(self):
        super(RecConv, self).__init__()
        #first use embedding layer to encode tokens
        self.embed = nn.Embedding(feature_dim, embed_dim)
        #convolution followed by pooling layer to reduce dimension, then
        #recurrent layer to retain sequential data.
        self.conv = nn.Conv1d(conv_channels_in, conv_channels_out, conv_kern_sz)
        self.pool = nn.MaxPool1d(pool_kern_sz)
        self.lstm = nn.LSTM(conv_lin_sz, hidden_dim)        
        self.lin = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    #Args: x is batched sequences of integers, each sequence is a sentence,
    #each integer represents a token.
    def forward(self, x):
        x = self.embed(x)
        x = F.relu(self.conv(x))
        x = self.pool(x)
        #need to reshape x at runtime, since don't know shape at model initialization.
        x = x.view(-1, conv_lin_sz)
        x = self.lstm(x)
        x = self.lin(x)
        return self.softmax(x)

model = RecConv()
opt = optim.SGD(model.parameters(), lr = 0.005, momentum = 0.9)
criterion = nn.NLLLoss()

'''
Args: x consists of sequences of integers, y consists of one-hot vectors representing
labels (i.e. 1 at label index, 0 elsewhere).
'''
def train(x, y):
    opt.zero_grad()
    predicted = model(x)
    loss = criterion(predicted, y)
    loss.backward()
    opt.step()
