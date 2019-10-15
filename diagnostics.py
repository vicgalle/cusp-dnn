import torch
import torch.nn as nn
import torch.nn.functional as F

def do_diagnostics(model, args):
    for name, param in model.named_parameters():
        if param.requires_grad and name.startswith('fc_inners'):
            print(name, torch.norm(param.data))
