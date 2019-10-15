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
    """ Samples as in Eq (2) from [1] """
    g1 = dist.Gamma(torch.tensor([a1]), torch.tensor([1.])).sample([1]).squeeze()
    g2 = dist.Gamma(torch.tensor([a1]), torch.tensor([1.])).sample([h-1]).squeeze()

    print(torch.cumprod( g2, dim=0))

    taus = torch.cat((g1, torch.cumprod( g2, dim=0)), 0)

    print(g1)
    print(g2)
    print(taus)

multiplicative_gamma(.2, .2, 5)











""" References

[1] - Durante, D. "A note on the multiplicative gamma process". 2016.

"""
