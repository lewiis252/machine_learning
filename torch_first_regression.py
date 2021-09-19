import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
sns.set()
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


# X = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
# y = X.pow(2) + 0.2*torch.rand(X.size())
X = torch.tensor([0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0]).to(device)
y = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]).to(device)


# plot data
plt.scatter(X.to('cpu'), y.to('cpu'))
plt.show()

X = torch.unsqueeze(X, dim=1)
y = torch.unsqueeze(y, dim=1)
# X = torch.tensor(X).to(device)
#
# y = torch.tensor(y).to(device)

'''Neural net'''
train_loss_list = []
val_loss_list = []
epoch_list = []

n_features = 1
hidden_neurons = 100
hidden_neurons2 = 50
# network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_features, hidden_neurons)  # hidden layer
        # self.hidden2 = torch.nn.Linear(hidden_neurons, hidden_neurons2)
        self.predict = torch.nn.Linear(hidden_neurons, 1)  # output layer

    def forward(self, x):
        out = F.selu(self.hidden(x))  # activation function for hidden layer
        # out = F.selu(self.hidden2(out))
        out = self.predict(out)  # linear output
        return out

def training_loop(n_epochs, optimizer, model, loss_fn, X, y):
    for epoch in range(1, n_epochs+1):


        outputs = model(X)
        loss = loss_fn(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        epoch_list.append(epoch)
        train_loss_list.append(loss)

        # set when to print info about training progress
        if epoch == 1 or epoch % 100 == 0:
            print('Epoch {}, Training loss {}'.format(epoch, np.sqrt(loss.to('cpu').detach().numpy())))

n_epochs = 250
model = Net().to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

# epoch_num_of_no_improve = 5

training_loop(n_epochs=n_epochs,
              optimizer=optimizer,
              model=model,
              loss_fn=loss_fn,
              X=X,
              y=y)

plt.figure(2)
plt.plot(epoch_list, train_loss_list, color='blue', label='train_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()


X_to_predict = torch.unsqueeze(torch.linspace(-5,30,10000), dim=1).to(device)
predict = model(X_to_predict)


plt.figure(3)
plt.scatter(X.to('cpu').detach().numpy(), y.to('cpu').detach().numpy())
plt.plot(X_to_predict.to('cpu').detach().numpy(), predict.to('cpu').detach().numpy())
plt.show()

