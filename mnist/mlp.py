
'''
Pytorch implementation of MLP training on MNIST.
'''
import torch
import torch.nn as nn

hidden_dim = 256
learning_rate = 0.005

class MLP(nn.Module):
    
    def __init__(self, x_dim, hidden_dim, y_dim):
        super(MLP, self).__init__()
        self.lin1 = nn.Linear(x_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, y_dim)
        self.dropout = nn.Dropout(0.3)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x1 = self.lin1(x)
        hidden = self.lin2(x1)
        x2 = self.dropout(hidden)
        out = self.softmax(x2)
        return out

mlp = MLP(len(x), hidden_dim, len(y))
opt = torch.optim.SGD(mlp.parameters(), lr = learning_rate)
       
#one training step    
def train(x, y):
    opt.zero_grad()
    #use log likelihood loss because last layer is log softmax
    criterion = nn.NLLLoss()
    predicted = mlp(x)
    loss = criterion(predicted, y)
    loss.backward()    
    opt.step()
