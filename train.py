import argparse
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models

from models import *
from diagnostics import do_diagnostics

# TODOs
# Fix hyperparameters to match previous literature

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', help="enable gpu training and inference",
                    action="store_true", default=True)

parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--res', help="enable residual connections",
                    action="store_true", default=False)
parser.add_argument('--n_res', type=int, default=1) ## Number of layers
parser.add_argument('--p_ber', type=float, default=0.1)
##
parser.add_argument('--a1', type=float, default=10.0) #Parameter of first gamma
parser.add_argument('--a2', type=float, default=10.0) #Parameter of second gamma
parser.add_argument('--l2', type=float, default=1e-4)
#parser.add_argument('-s', '--samples', type=int, default=1)
parser.add_argument('--hid_dim', type=int, default=50)

parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--seed', type=int, default=0)

ds = ['mnist', 'cifar']
parser.add_argument('--dataset', choices=ds, default='mnist')

noises = ['none', 'bernoulli', 'cumulative_bern', 'decay_gauss', 'addexp', 'addgamm', 'cumgamm']
parser.add_argument('--noise', choices=noises, default='none')


args = parser.parse_args()
# Set the random seed, so the experiment is reproducible
torch.manual_seed(args.seed)
# For the moment, we will just train on CPU, so no cuda
use_cuda = args.gpu
device = torch.device("cuda" if use_cuda else "cpu")


def train(model, device, train_loader, optimizer, epoch, train_losses, criterion):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        #loss = F.nll_loss(output, target)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            #test_loss += F.nll_loss(output, target, reduction='sum').item()
            test_loss += criterion(output, target).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if args.dataset == 'mnist':
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True, 
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, num_workers=2)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, num_workers=2)

    # The size of the input. MNIST are greyscale images, 28x28 pixels each
    in_size = 28*28
    out_dim = 10

elif args.dataset == 'cifar':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(root='../data', train=True,
                                            download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root='../data', train=False,
                                        download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,
                                            shuffle=False, num_workers=2)

    in_size = 32*32*3
    out_dim = 10

if args.noise == 'none':
    def dropout(x, context): return x

elif args.noise == 'bernoulli':
    dropout = Dropout(p=args.p_ber).to(device)

elif args.noise == 'cumulative_bern':
    dropout = CumulativeDropout().to(device)

elif args.noise == 'addexp':
    dropout = GammaProcesses('exp', args.a1, args.a2, args.n_res)

elif args.noise == 'addgamm':
    dropout = GammaProcesses('add', args.a1, args.a2, args.n_res)

elif args.noise == 'cumgamm':
    dropout = GammaProcesses('mul', args.a1, args.a2, args.n_res)

elif args.noise == 'decay_gauss':
    dropout = ExpDecayGauss().to(device)

model = MLP(in_size, out_dim, args.hid_dim, dropout,  args).to(device)

model = models.resnet18(pretrained=False).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer,  milestones=[150, 250], gamma=0.1)

print(model)

training_losses = []
for epoch in range(1, args.epochs + 1):
    t0 = time.time()
    train(model, device, train_loader, optimizer, epoch, training_losses, criterion)
    t1 = time.time()
    print('Epoch ', epoch, '\tdt = ', t1 - t0)

    test(model, device, test_loader, criterion)

    scheduler.step()

do_diagnostics(model, args)
