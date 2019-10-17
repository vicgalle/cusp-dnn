import torch
import torch.distributions as dist


def stick_breaking(alpha0, k):
    """ This function breaks a stick into k pieces """
    betas = dist.Beta(torch.tensor([1.]), torch.tensor([alpha0])).sample([k]).squeeze()
    remains = torch.cat((torch.tensor([1.]), torch.cumprod( 1 - betas[:-1], dim=0)), 0)
    p = betas * remains
    p /= p.sum()
    return p

def multiplicative_gamma(a1, a2, h):
    """
    Samples as in Eq (2) from [1]
    h is the number of layers
    """
    g1 = dist.Gamma(torch.tensor([a1]), torch.tensor([1.])).sample([1]).squeeze(0)
    g2 = dist.Gamma(torch.tensor([a2]), torch.tensor([1.])).sample([h-1]).squeeze()
    taus = torch.cat( (g1, g1*torch.cumprod(g2, dim=0) ) , 0)
    ##
    return taus

def additive_gamma(a1, a2, h):
    """
    Samples from additive gamma process
    h is the number of layers
    """
    g1 = dist.Gamma(torch.tensor([a1]), torch.tensor([1.])).sample([1]).squeeze(0)
    g2 = dist.Gamma(torch.tensor([a2]), torch.tensor([1.])).sample([h-1]).squeeze()
    taus = torch.cat( (g1, g1 + torch.cumsum(g2, dim=0) ) , 0)
    ##
    return taus

def additive_exponential(alpha1, alpha2, h):
    """
    Samples from additive exponential process
    h is the number of layers
    alpha is the rate of the exponential distribution
    """
    g1 = dist.Exponential(torch.tensor([alpha1])).sample([1]).squeeze(0)
    g2 = dist.Exponential(torch.tensor([alpha2])).sample([h-1]).squeeze()
    taus = torch.cat( (g1, g1 + torch.cumsum(g2, dim=0) ) , 0)
    ##
    return taus

additive_exponential(1., 22., 5)

""" References

[1] - Durante, D. "A note on the multiplicative gamma process". 2016.

"""
