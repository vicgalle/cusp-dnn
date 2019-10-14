import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, residual=True):
        super(MLP, self).__init__()
        self.d_in = input_dim
        self.residual = residual
        self.fc1 = nn.Linear(self.d_in, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.d_in)
        x = F.relu(self.fc1(x))

        if self.residual:
            x = F.relu(self.fc2(x)) + x
        else:
            x = F.relu(self.fc2(x))
        
        x = self.fc3(x)
        return F.log_softmax(x)

