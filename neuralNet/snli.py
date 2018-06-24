
'''
Nets to classify snli, of increasing complexity.

'''

import torch.nn as nn
import torch.functional as F
import torch.optim as optim

#size of list of most frequently used words
EMBED_IN =
#embed dim
EMBED_DIM = 300
LIN_DIM = 50

class SNLI0(nn.module):
    
    def __init__(self, embed_in_sz, embed_dim):
        super(SNLI0, self).__init__()
        self.embed_dim = embed_dim
        #trainable embeddings
        self.embed = nn.Embedding(embed_in_sz, embed_dim)
        #2 times because concatenating hyp and text
        self.lin1 = nn.Linear(2 * embed_dim, LIN_DIM)
        self.softmax = nn.Softmax()

    #hyp and text are both batches of sequences of word indices
    def forward(self, hyp, text):
        #embed both hyp and text sequences
        hyp_embed = self.embed(hyp)
        text_embed = self.embed(text)
        #turn the sequence embeddings into a single fixed-length vector,
        #apply tanh activation to each word embedding
        hyp_size = hyp_embed.size()
        hyp_embed = F.tanh(hyp_embed.view(-1, self.embed_dim)).view(hyp_size[0], hyp_size[1], -1)
        text_size = text_embed.size()
        text_embed = F.tanh(text_embed.view(-1, self.embed_dim)).view(text_size[0], text_size[1], -1)
        
        #combine the vectors by taking average of hyp and text sequence embeddings        
        hyp_embed = F.mean(hyp_embed, 1)
        text_embed = F.mean(text_embed, 1)
        #concatenate hyp and text
        combined = torch.cat((hyp_embed, text_embed))
        combined = F.relu(self.lin1(combined))
        #softmax over the three categories: E, N, C
        out = self.softmax(out)
        return out
    
'''
Use pretrained embeddings, e.g. GLOVE, and and LSTM to combine vectors.
'''
class SNLI1(nn.module):
    
    def __init__(self, embed_dim, hidden_dim, embed_weights):
        super(SNLI0, self).__init__()
        self.embed_dim = embed_dim
        self.rnn_hidden_dim = hidden_dim
        #pretrained embeddings, weights frozen by default
        self.embed = nn.Embedding.from_pretrained(embed_weights)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        #2 times because concatenating hyp and text
        self.lin1 = nn.Linear(2 * hidden_dim, LIN_DIM)
        self.softmax = nn.Softmax()
        self.hyp_hidden = init_hidden()
        self.text_hidden = init_hidden()

    #hyp and text are both batches of sequences of word indices
    def forward(self, hyp, text):
        #embed both hyp and text sequences
        hyp_embed = self.embed(hyp)
        text_embed = self.embed(text)
        #turn the sequence embeddings into a single fixed-length vector,
        #apply tanh activation to each word embedding
        hyp_size = hyp_embed.size()
        hyp_embed = F.tanh(hyp_embed.view(-1, self.embed_dim)).view(hyp_size[0], hyp_size[1], -1)
        text_size = text_embed.size()
        text_embed = F.tanh(text_embed.view(-1, self.embed_dim)).view(text_size[0], text_size[1], -1)
        
        #combine the vectors by taking average of hyp and text sequence embeddings        
        hyp_embed, self.hyp_hidden = self.lstm(hyp_embed, self.hyp_hidden)
        text_embed, self.text_hidden = self.lstm(text_embed, self.text_hidden)
        #concatenate hyp and text
        combined = torch.cat((hyp_embed, text_embed))
        combined = F.relu(self.lin1(combined))
        #softmax over the three categories: E, N, C
        out = self.softmax(out)
        return out
    
    def init_hidden(self):
        return (torch.zeros(1, 1, self.rnn_hidden_dim),
                torch.zeros(1, 1, self.rnn_hidden_dim))

'''
load pretrained vectors, each line of form "word v1 v2 ..."
Returns:
-dictionary of word key and np embedding vector.
'''
def load_pretrained(path):
    word_to_vec = {}
    with open(path, "r") as file:
        for line in file:
            line_ar = line.split()
            word = line_ar[0]
            embed = np.asarray(line_ar[1:], dtype='float32')
            word_to_vec[word] = embed
    return word_to_vec

