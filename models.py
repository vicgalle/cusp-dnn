import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim, other_args):
        super(MLP, self).__init__()
        self.d_in = in_dim
        self.residual = other_args.res
        self.n_res = other_args.n_res

        self.fc_in = nn.Linear(self.d_in, hid_dim)
        self.fc_inners = nn.ModuleList([nn.Linear(hid_dim, hid_dim) for _ in range(self.n_res)])
        self.fc_out = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        x = x.view(-1, self.d_in)
        x = F.relu(self.fc_in(x))

        for fc_inner in self.fc_inners:
            if self.residual:
                x = F.relu(fc_inner(x)) + x
            else:
                x = F.relu(fc_inner(x))
        
        x = self.fc_out(x)
        return F.log_softmax(x)

