
'''
Sequence-to-sequence model to learn addition of two-digit 
positive natural numbers.
'''
import torch
import torch.nn as nn
import torch.functional as F
import numpy as np

class AddNet(nn.Module):
    def __init__(self, nin, nhidden, nout):
        super(AddNet, self)__init__()
        self.nhidden = nhidden
        self.lstm = nn.LSTM(nin, nhidden, batch_first=True)
        self.lin = nn.Linear(nhidden, nout)

    def forward(self, x, y, xhidden, yhidden):
        x_seq = to_digits(x)
        y_seq = to_digits(y)
        x_seq, xhidden = self.lstm(x_seq, xhidden)
        y_seq, yhidden = self.lstm(y_seq, yhidden)
        
        out = torch.cat((x_seq, y_seq), dim = 0)
        out = self.lin(out)
        return out, xhidden, yhidden

    def to_digits(self, x):
        digits = []
        while x > 0:
            diff = x - x/10*10
            digits.append(diff)
            x /= 10
        digits = np.array(digits.reverse())
        ar = np.zeros(len(digits), 10)
        ar[np.arange(len(digits)), digits] = 1
        return ar


