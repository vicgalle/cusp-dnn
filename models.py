import torch
import torch.nn as nn
import torch.nn.functional as F
import utils as ut

# check what to do with the biases of the linear layers. How do they do in Gal et al?


class MLP(nn.Module):
    """ A standard feedforward neural network with options for residual connections and dropouts """

    def __init__(self, in_dim, out_dim, hid_dim, dropout, other_args):
        super(MLP, self).__init__()
        self.d_in = in_dim
        self.residual = other_args.res
        self.n_res = other_args.n_res
        self.dropout = dropout

        self.fc_in = nn.Linear(self.d_in, hid_dim)
        self.fc_inners = nn.ModuleList(
            [nn.Linear(hid_dim, hid_dim, bias=False) for _ in range(self.n_res)])
        self.fc_out = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        x = x.view(-1, self.d_in)
        x = F.relu(self.fc_in(x))

        for l, fc_inner in enumerate(self.fc_inners):

            x_d = self.dropout(x, l)

            if self.residual:
                x = F.relu(fc_inner(x_d)) + x
            else:
                x = F.relu(fc_inner(x_d))

        x = self.fc_out(x)
        return F.log_softmax(x)


class Dropout(nn.Module):
    """ This module adds (standard Bernoulli) Dropout to the following weights of a layer.
    """

    def __init__(self, p=0.1):
        super(Dropout, self).__init__()
        assert p <= 1.
        assert p >= 0.
        self.p = p

    def forward(self, x, context=None):
        if self.training:
            binomial = torch.distributions.binomial.Binomial(
                probs=torch.tensor(1-self.p, device=x.device))
            x = x * binomial.sample(x.size()) * \
                (1. / (1. - self.p))   # inverted dropout
        return x


class CumulativeDropout(nn.Module):
    """ This module adds cumulative (standard Bernoulli) Dropout to the following weights of a layer.
    """

    def __init__(self, p=0.1, step=0.1):
        super(CumulativeDropout, self).__init__()
        assert p <= 1.
        assert p >= 0.
        self.p = p
        self.step = step

    def forward(self, x, context=0):
        # We increase the dropout amount with the context (represents the layer)
        if self.training:
            p = self.p + context*self.step
            binomial = torch.distributions.binomial.Binomial(
                probs=torch.tensor(1-p, device=x.device))
            x = x * binomial.sample(x.size()) * \
                (1. / (1. - p))   # inverted dropout
        return x

class GammaProcesses(nn.Module):
    """ This module implements the additive gamma "dropout"
    * L is an int indicating the number of layersself.
    * typ is a string with possible values "exp", "mul", "add", indicating
    type of prior (exponential, additive gamma, multiplicative gamma)
    * a1 and a2 are the two parameters of the gammas/exponentials
    """

    def __init__(self, a1=3.0, a2=4.0, L=5, typ="exp"):
        super(GammaProcesses, self).__init__()
        self.typ = typ
        self.L = L
        self.a1 = a1
        self.a2 = a2
        ## Initialize the probs
        if self.typ == "exp":
            self.pp = ut.additive_exponential(self.a1, self.a2, self.L)
        elif self.typ == "add":
            self.pp = ut.additive_gamma(self.a1, self.a2, self.L)
        elif self.typ == "mul":
            self.pp = ut.multiplicative_gamma(self.a1, self.a2, self.L)

    def forward(self, x, context=0):
        # Context is the index of the layer
        if self.training:
            x = x * self.pp
        return x
