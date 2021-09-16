from matplotlib import pyplot as plt
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# class_names = ['airplane','automobile','bird','cat','deer',
#                'dog','frog','horse','ship','truck']

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

# load data, no think
cifar10_train = datasets.CIFAR10(
    root='data', train=True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                             (0.2470, 0.2435, 0.2616))]))
train_length = len(cifar10_train)
train_size = int(0.8 *train_length)
val_size = train_length - train_size

# make trai and validation set
cifar10_train, cifar10_val = torch.utils.data.random_split(cifar10_train, [train_size, val_size])

cifar10_test = datasets.CIFAR10(
    root='data', train=False, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                             (0.2470, 0.2435, 0.2616))]))

# comment this and change output neurons (and dataloader far below) if you want only to find difference beetwenn planes and birds
# get only birds and planes
label_map = {0: 0, 2: 1}
class_names = ['airplane', 'bird']
cifar10_train_ = [(img, label_map[label])
          for img, label in cifar10_train
          if label in [0, 2]]

cifar10_val_ = [(img, label_map[label])
          for img, label in cifar10_val
          if label in [0, 2]]

cifar10_test_ = [(img, label_map[label])
              for img, label in cifar10_test
              if label in [0, 2]]

# store train and val loss
train_loss_list = []
val_loss_list = []
epoch_list = []

# make network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1) # convolution layer (in_chl, out_chl,...)
        self.act1 = nn.Tanh() # activation function
        self.pool1 = nn.MaxPool2d(2) # pooling (kernel size 2x2)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.act2 = nn.Tanh()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(8 * 8 * 8, 32) # first 8 from conv2, next 8's from pooling (32->16->8)
        self.act3 = nn.Tanh()
        self.fc2 = nn.Linear(32, 2)


    def forward(self, x):
        out = self.pool1(self.act1(self.conv1(x)))
        out = self.pool2(self.act2(self.conv2(out)))
        out = out.view(-1, 8 * 8 * 8)  # not sure why reshape
        out = self.act3(self.fc1(out))
        out = self.fc2(out)
        return out

import datetime # to measure time

def training_loop(n_epochs, optimizer, model, loss_fn, train_loader):
    for epoch in range(1, n_epochs+1):
        loss_train = 0.0
        for imgs, labels in train_loader:

            # move tensors to gpu if available
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)

            outputs = model(imgs)

            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            loss_train += loss.item()

        if epoch == 1 or epoch % 1 == 0:
            epoch_list.append(epoch)
            train_loss_list.append(loss_train / len(train_loader)) # to track loss
            # get loss of validation data
            with torch.no_grad():
                loss_val = 0.0
                for imgs, labels in val_loader:
                    # move tensors to gpu if available
                    imgs = imgs.to(device=device)
                    labels = labels.to(device=device)

                    outputs = model(imgs)

                    loss_v = loss_fn(outputs, labels)

                    loss_val += loss_v.item()
            val_loss_list.append(loss_val / len(val_loader))

            print('{} Epoch {}, Training loss {}, Validation loss {}'.format(
                datetime.datetime.now(), epoch,
                loss_train / len(train_loader), loss_val/len(val_loader)))

def validate_on_test(model, train_loader, test_loader):
    for name, loader in [("train", train_loader), ("val", val_loader), ('test', test_loader)]:
        correct = 0
        total = 0

        with torch.no_grad():  # <1>
            for imgs, labels in loader:

                # move to gpu
                imgs = imgs.to(device=device)
                labels = labels.to(device=device)

                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1) # Gives us the index of the highest value
                total += labels.shape[0]  # Counts the number of examples, so total is increased by the batch size
                correct += int((predicted == labels).sum())

        print("Accuracy {}: {:.2f} %".format(name , 100 *  (correct / total)))

n_epochs = 100
model = Net().to(device=device)
optimizer = optim.SGD(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()
train_loader = torch.utils.data.DataLoader(cifar10_train_, batch_size=64, shuffle=False)
val_loader = torch.utils.data.DataLoader(cifar10_val_, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(cifar10_test_,  batch_size=64, shuffle=False)

training_loop(
    n_epochs = n_epochs,
    optimizer = optimizer,
    model = model,
    loss_fn = loss_fn,
    train_loader = train_loader)

validate_on_test(model, train_loader, test_loader)

plt.plot(epoch_list, train_loss_list, color='blue', label='train_loss')
plt.plot(epoch_list, val_loss_list, color='green', label='validation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()