from torchvision import datasets
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn as nn

data_path = 'datasets'
cifar10_train = datasets.CIFAR10(data_path,
                                 train=True,
                                 download=True,
                                 transform=transforms.ToTensor())
cifar10_val = datasets.CIFAR10(data_path,
                               train=False,
                               download=True,
                               transform=transforms.ToTensor())

print(len(cifar10_train))

# plot sample
img, label = cifar10_train[99]
plt.imshow(img.permute(1,2,0)) #Changes the order of the axes from C × H × W to H × W × C
plt.show()

print('Sample shape', img.shape)

# normalizing data
imgs = torch.stack([img_t for img_t, _ in cifar10_train], dim=3)
print(imgs.shape)

calculed_mean = imgs.view(3, -1).mean(dim=1) # Recall that view(3, -1) keeps the three channels and merges all the
# remaining dimensions into one, figuring out the appropriate size. Here our 3 × 32 × 32 image is transformed into a
# 3 × 1,024 vector, and then the mean is taken over the 1,024 elements of each channel.
calculed_std = imgs.view(3, -1).std(dim=1)

transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616))

transformed_cifar10 = datasets.CIFAR10(data_path,
                                       train=True,
                                       download=False,
                                       transform=transforms.Compose([transforms.ToTensor(),
                                                                     transforms.Normalize((calculed_mean),
                                                                                          (calculed_std))]))

label_map = {0: 0, 2: 1}
class_names = ['airplane', 'bird']
cifar2_train = [(img, label_map[label])
          for img, label in cifar10_train
          if label in [0, 2]]

cifar2_val = [(img, label_map[label])
              for img, label in cifar10_val
              if label in [0, 2]]

n_out = 2
print('Labels selected')

train_loader = torch.utils.data.DataLoader(cifar2_train, batch_size=64,
shuffle=True)

model = nn.Sequential(
    nn.Linear(32*32*3,512), # input features and hidden layer size
    nn.ReLU(), # hidden layer activation
    nn.Linear(512,n_out), # output
    nn.LogSoftmax(dim=1)# output activation
)

# initiate loss, learning rate and optimizer
loss_fn = nn.NLLLoss()
learning_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_epochs = 10

for epoch in range(n_epochs):
    for imgs, labels in train_loader:
        batch_size = imgs.shape[0]
        outputs = model(imgs.view(batch_size, -1)) # 1D tensor with extra dimension
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch: %d, Loss: %f' % (epoch, float(loss)))

val_loader = val_loader = torch.utils.data.DataLoader(cifar2_val, batch_size=64, shuffle=False)

correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in val_loader:
        batch_size = imgs.shape[0]
        outputs = model(imgs.view(batch_size, -1))
        _, predicted = torch.max(outputs, dim=1)
        total += labels.shape[0]
        correct += int((predicted == labels).sum())
    print("Accuracy:", correct / total)





