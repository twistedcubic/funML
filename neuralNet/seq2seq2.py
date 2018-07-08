

'''
Sequence-to-sequence, text to text.

Uses encoder and decoder with attention
'''


N_EMBED = 32
N_HIDDEN = 50
N_SEQ = 20

class Decoder(nn.Module):
    def __init__(self, n_hidden, n_out):
        super(self, Decoder).__init__()
        self.attn = nn.Linear(nhidden+N_EMBED, N_SEQ)
        self.lin1 = nn.Linear(nhidden+N_EMBED, nhidden)
        #number of input feature dimension, hidden units, hidden layer dimension
        self.gru = nn.GRU(2*n_hidden, n_hidden)
        self.lin2 = nn.Linear(n_hidden, n_out)
        
    def init_hidden(self,  n_hidden, n_batch=1, n_seq=1):
        #need batch dimension, embed dim, and sentence len
        #batch_first is False by default
        return torch.randn(n_seq, b_batch, n_hidden)
    
    '''
    need previous hidden state, encoder final state, and next word (its embedding),
    w should have dim (batch, embed_dim). 
    '''
    def forward(self, hidden, enc_state, w):
        w = w.unsqueeze(1)
        w2 = self.attn(torch.cat((w, hidden), dim=2))
        w2 = F.softmax(w2)
        
        #w2 has shape (batch_sz, 1, n_seq), and enc_state (n_seq, batch_sz, n_hidden)
        #combined has dim (batch_sz, 1, n_hidden)
        combined = F.bmm(w2, enc_state.permute(1, 0, 2))
        #w has shape (batch_sz, 1, N_EMBED), combined has dim (batch_sz, 1, n_hidden)
        combined = self.lin1(torch.cat((w, combined), dim=2))
        #this has dim (batch, 1, n_hidden)
        combined = F.relu(combined)
        out, hidden = self.gru(combined, hidden)
        out = F.softmax(self.lin2(out))
        return out, hidden
