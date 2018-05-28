
'''
Pytorch CNN model classifying texts.
This is the standard alternating layer structure with 
convolution, pooling, and dropout, following an embedding
layer to encode the input characters.
'''
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

kernel_sz = 3
embed_dim = 128
dict_dim = 4000
label_dim = 16
conv_channel_in = 3
conv_channel_out = 3
conv_kernel_sz = 3
pool_kernel_sz = 5
conv_lin_dim = 8

learning_rate = 0.001
momentum = 0.9
train_steps = 1000

class MyCNN(nn.Model):
    def __init__(self, input_dim, out_dim):
        super.__init__(MyCNN, self)
        self.embed = nn.Embedding(input_dim, embed_dim)
        self.conv = nn.Conv1d(conv_channel_in, conv_channel_out, conv_kernel_sz)
        self.pool = nn.MaxPool1d(pool_kernel_sz)
        self.lin = nn.Linear(conv_lin_dim, out_dim)
        self.dropout = nn.Dropout(0.3)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input):
        x = F.relu(self.conv(input))
        x = self.pool(x)
        out = x.view(-1, conv_lin_dim)
        out = self.lin(out)
        out = self.dropout(out)
        out = self.softmax(out)
        return out

criterion = nn.NLLLoss()
myCNN = MyCNN(dict_dim, label_dim)
opt = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    
def train(x, y):
    for i in range(train_steps):
        opt.zero_grad()
        predicted = myCNN(x)
        loss = criterion(predicted, y)
        loss.backward()
        opt.step()
