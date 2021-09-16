from matplotlib import pyplot as plt
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

# not needed AFTER getting mean and standard deviation
# cifar10_train = datasets.CIFAR10(
#     root='data', train=True, download=True,
#     transform=transforms.ToTensor())
#
# cifar10_val = datasets.CIFAR10(
#     root='data', train=False, download=True,
#     transform=transforms.ToTensor())

# imgs_train = torch.stack([img_t for img_t, _ in cifar10_train], dim=3)
# imgs_val = torch.stack([img_t for img_t, _ in cifar10_val], dim=3)

# train_mean = imgs_train.view(3,-1).mean(dim=1)
# train_std = imgs_train.view(3,-1).std(dim=1)
#
# val_mean = imgs_val.view(3,-1).mean(dim=1)
# val_std = imgs_val.view(3,-1).std(dim=1)

cifar10_train = datasets.CIFAR10(
    root='data', train=True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                             (0.2470, 0.2435, 0.2616))]))

cifar10_val = datasets.CIFAR10(
    root='data', train=False, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                             (0.2470, 0.2435, 0.2616))]))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out = out.view(-1, 8 * 8 * 8)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out

import datetime

def training_loop(n_epochs, optimizer, model, loss_fn, train_loader):
    for epoch in range(1, n_epochs+1):
        loss_train = 0.0
        for imgs, labels in train_loader:

            imgs = imgs.to(device=device)
            labels = labels.to(device=device)

            outputs = model(imgs)

            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            loss_train += loss.item()

        if epoch == 1 or epoch % 10 == 0:
            print('{} Epoch {}, Training loss {}'.format(
                datetime.datetime.now(), epoch,
                loss_train / len(train_loader)))

def validate(model, train_loader, val_loader):
    for name, loader in [("train", train_loader), ("val", val_loader)]:
        correct = 0
        total = 0

        with torch.no_grad():  # <1>
            for imgs, labels in loader:
                imgs = imgs.to(device=device)
                labels = labels.to(device=device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1) # <2>
                total += labels.shape[0]  # <3>
                correct += int((predicted == labels).sum())  # <4>

        print("Accuracy {}: {:.2f}".format(name , correct / total))






n_epochs = 100
model = Net().to(device=device)
optimizer = optim.SGD(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()
train_loader = torch.utils.data.DataLoader(cifar10_train, batch_size=64, shuffle=False)
val_loader = torch.utils.data.DataLoader(cifar10_val, batch_size=64, shuffle=False)

training_loop(
    n_epochs = n_epochs,
    optimizer = optimizer,
    model = model,
    loss_fn = loss_fn,
    train_loader = train_loader)

validate(model, train_loader, val_loader)
