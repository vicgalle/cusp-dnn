import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import os


def do_diagnostics(model, args):
    directory = 'viz/'
    for name, param in model.named_parameters():
        if param.requires_grad and name.startswith('fc_inners'):
            print(name, torch.norm(param.data))
            print(name, torch.norm(param.data, dim=0))
            print(name, torch.norm(param.data, dim=1))

            if not os.path.exists(directory):
                os.makedirs(directory)

            plt.matshow(param.data.cpu())
            plt.colorbar()
            plt.savefig(directory + name + '_' + args.noise +'.png')
