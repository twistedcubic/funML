
'''
Net for generating texts in a certain style, 
given training texts. Uses a recurrent layer
to remember the sequential data.
'''

import torch
import torch.nn as nn
import torch.functional as F
import numpy as np

class TextGen(nn.Module):
    
    #nx is input feature space dimension
    def __init__(self, nx, nhidden, ny):
        super(TextGen, self).__init__()
        self.embed = nn.Embed(nx, nhidden)
        self.rnn = nn.LSTM(nhidden, nhidden)
        self.lin = nn.Linear(2*nhidden, ny)

    #x is batch of word indices, each index represents the word
    #following the corresponding word in prev batch.
    def forward(self, x, hidden):
        x = self.embed(x)
        x = self.rnn(x)
        #concat current input char and previous hidden state
        #to predict next char
        x = F.relu(self.lin(torch.cat(x, hidden)))
        x = F.log_softmax(x, dim=1)
        return x, hidden

    def init_hidden(self, nlayer, nbatch, nhidden):
        return (torch.zeros(nlayer, nbatch, nhidden),
                torch.zeros(nlayer, nbatch, nhidden))

TXT_FEATURES = 3000
HIDDEN_DIM = 256
BATCH_SZ = 32
NUM_LAYERS = 1
#max allowed number of tokens in a sentence
MAX_SEN_LEN = 20

model = TextGen(TXT_FEATURES, HIDDEN_DIM, TXT_FEATURES)
opt = optim.SGD(model.parameters())
criterion = nn.NLLLoss()

#Data consist of a batch of padded sequences, dim 0 is a slice of all
#text data, and dim 1 consists of the different data samples.
def train(data):    
    hidden = model.init_hidden(NUM_LAYERS, BATCH_SZ, HIDDEN_DIM)    
    for i in range(data.size(0)-1):
        text_slice = data[i]
        opt.zero_grad()
        pred = model(text_slice, hidden)
        loss = criterion(pred, data[i+1])
        loss.backward()
        opt.step()    

#Create the two dictionaries for lookup and reverse lookup
#Args: -list of words from tokenized text.
def create_maps(text):
    text = [w for sublist in text for w in sublist]
    text_set = set(text)
    word_to_index = {w : i for (i, w) in enumerate(text_set)}
    #for reverse look up
    index_to_word = {i : w for (i, w) in enumerate(text_set)}
    return word_to_index, index_to_word

'''
Args:
-batches of list of tokens in training text.
'''
def process_and_train(text):
    
    word_to_index, index_to_word = create_maps(text)    
    data = np.zeros((MAX_SEN_LEN, len(text)))
    for i, l in enumerate(text):
        #rows are slices in text, and columns are
        #text samples
        
        for j, token in enumerate(l):
            if j >= MAX_SEN_LEN:
                break
            try:
                data[j][i] = word_to_index(token)
            except KeyError:
                pass
    train(torch.FloatTensor(data))
    
    
    
