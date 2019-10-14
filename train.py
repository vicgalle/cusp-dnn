import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from models import MLP

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', help="enable gpu training and inference",
                    action="store_true", default=True)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.5)
parser.add_argument('--res', help="enable residual connections",
                    action="store_true", default=False)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--seed', type=int, default=0)

ds = ['mnist']
parser.add_argument('--dataset', choices=ds, default='mnist')


args = parser.parse_args()
# Set the random seed, so the experiment is reproducible
torch.manual_seed(args.seed)
# For the moment, we will just train on CPU, so no cuda
use_cuda = args.gpu
device = torch.device("cuda" if use_cuda else "cpu")


def train(model, device, train_loader, optimizer, epoch, train_losses):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if args.dataset=='mnist':
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=args.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True)

    # The size of the input. MNIST are greyscale images, 28x28 pixels each
    in_size = 28*28

model = MLP(in_size, args.res).to(device)
optimizer = optim.SGD(model.parameters(), lr=args.lr,
                      momentum=args.momentum)

training_losses = []
for epoch in range(1, args.epochs + 1):
    train(model, device, train_loader, optimizer, epoch, training_losses)
    test(model, device, test_loader)
