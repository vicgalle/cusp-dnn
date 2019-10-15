import torch
import torch.distributions as dist


def stick_breaking(alpha0, k):
    """ This function breaks a stick into k pieces """ 
    betas = dist.Beta(torch.tensor([1.]), torch.tensor([alpha0])).sample([k]).squeeze()
    remains = torch.cat((torch.tensor([1.]), torch.cumprod( 1 - betas[:-1], dim=0)), 0)
    p = betas * remains
    p /= p.sum()
    return p
