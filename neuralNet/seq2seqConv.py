
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
    def __init__(self, Encode):
        super(Encode, self).__init__()
        
