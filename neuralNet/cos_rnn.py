
'''
Recurrent sequence-to-sequence model to estimate y=cosine(x).
Encodes the independent variable data into sequence of integers,
where the integer n denotes the nth multiple of pi/100. Where
the range of x is [0, 2*pi]. The range of y is [-1, 1], with
0.01 interval.
This unit is chosen, rather than decimals, because cosine is 
naturally periodic in pi.
'''
import torch.nn as nn
import nn.functional as F
import torch.optim as optim

HIDDEN_DIM = 100
BATCH_SZ = 64

class CosEncoder(nn.Module):
    def __init__(self, nx, nhidden):
        super(CosEncoder, self).__init__()
        self.nhidden = nhidden
        self.rnn = nn.LSTM(nx, nhidden)
        #hidden to be same
        self.lin = nn.Linear(nhidden, nhidden)
        
    def forward(self, x, hidden):        
        x, hidden = self.rnn(x, hidden)
        enc_out = self.lin(x)
        return enc_out, hidden

class CosDecoder(nn.Module):
    def __init__(self, nhidden, ny):
        super(CosDecoder, self).__init__()
        self.rnn = nn.LSTM(nhidden, nhidden)
        self.lin = nn.Linear(nhidden, ny)
        
    #initial hidden state will be the output state of the encoder.
    def forward(self, y, hidden):
        y, hidden = self.rnn(y, hidden)
        out = self.lin(y)
        return out, hidden

def init_hidden(nhidden, nbatch, nlen=1, nlayers=1):
    #initialize hidden state and cell state vectors
    return (torch.zeros(nlen*nlayers, nbatch, nhidden),
            torch.zeros(nlen*nlayers, nbatch, nhidden))
    
def train(input, output):
    encoder = CosEncoder( HIDDEN_DIM, input.size(-1))
    decoder = CosDecoder(HIDDEN_DIM, output.size(-1))
    
    encoder_opt = optim.SGD(encoder.parameters())
    decoder_opt = optim.SGD(decoder.parameters())

    for i in range(input.size(0)//BATCH_SZ):
        encoder.zero_grad()
        decoer.zero_grad()
        enc_hidden = init_hidden(HIDDEN_DIM, BATCH_SZ, nlen=input.size(1))
        #don't use enc_out, only encoder hidden state
        enc_out, enc_hidden = encoder(input[i*BATCH_SZ : min(input.size(0), (i+1)*BATCH_SZ)], enc_hidden)
        dec_hidden = init_hidden(HIDDEN_DIM, BATCH_SZ, nlen=output.size(1))
        dec_out, dec_hidden = decoder(output[i*BATCH_SZ : min(output.size(0), (i+1)*BATCH_SZ)], dec_hidden)
        
'''
Decoder with attention.
Attention exemplified as in peeking at the encoder output,
so that not all encoder input information rests within a single
context vector that is the final encoder hidden state.    
Although admittedly attention is overkill for this task.
'''
class CosAttnDecoder(nn.Module):
    
    def __init__(self, nhidden, ny):
        super(CosAttnDecoder, self).__init__()
        #ny+nhidden after concatenating prev hidden state
        #and decoder input y.
        self.attn_lin = nn.Linear(ny+nhidden, nhidden)
        self.softmax = nn.Softmax(dim=1)        
        self.rnn = nn.LSTM(nhidden, nhidden)
        self.lin = nn.Linear(nhidden, ny)
        
    #Initial hidden state will be the output state of the encoder.
    def forward(self, y, hidden, enc_output):
        y = self.attn_lin(torch.cat((y, hidden), dim=1))
        y = self.softmax(y)
        y = torch.bmm(y, enc_output)
        y = F.relu(self.lin(y))
        out, hidden = self.rnn(torch.cat((y, hidden), dim=1), hidden)
        #linear output, takes 
        out = self.lin(out)
        return out, hidden

'''
Non-recurrent model for cosine(x).
'''
class CosLinear(nn.Module):
    def __init__(self, nx, ny):        
        self.lin1 = nn.Linear(nx, 64)
        self.lin2 = nn.Linear(64, 32)
        #ny will be e.g. 201, for 201 0.01 interval end points
        #between -1 and 1. 
        self.lin3 = nn.Linear(32, ny)
        
                
    def forward(self, x):
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)
        #log softmax to be faster and more numerically-stable
        #than log(softmax(x))
        out = F.log_softmax(x)
        return out
