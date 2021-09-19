import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
sns.set()
from pandas.plotting import scatter_matrix

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("data", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()



def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

def get_data_info():
    print(housing.info())
    data_describe = housing.describe()
    return data_describe


fetch_housing_data()
housing = load_housing_data()
housing = housing.fillna(method='ffill')
describe = get_data_info()


housing['income per pop'] = housing['median_income'] / housing['population']
housing['room per pop'] = housing['total_rooms'] / housing['population']
housing['bedroms per pop'] = housing['total_bedrooms'] / housing['population']
housing["population_per_household"]=housing["population"]/housing["households"]

# plt.figure(1)
# housing.hist()
# plt.show()
#
# plt.figure(2)
# sns.pairplot(data=housing, y_vars='median_house_value')
# plt.show()
#
# plt.figure(3)
# sns.scatterplot(data=housing, x='longitude', y='latitude', hue='median_house_value', s=1)
# plt.show()


housing = pd.get_dummies(housing)

describe = get_data_info()
correlations = housing.corr()
scaler = MinMaxScaler()



y = housing.pop('median_house_value')
# scaling
# housing[housing.columns] = scaler.fit_transform(housing[housing.columns])
X = housing


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=42)

#scaling
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)


'''Neural net'''
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# convert a df to tensor to be used in pytorch

X_train = torch.from_numpy(X_train).float().to(device)
y_train = torch.unsqueeze(torch.from_numpy(y_train.values).float().to(device), dim=1)
X_test = torch.from_numpy(X_test).float().to(device)
y_test = torch.unsqueeze(torch.from_numpy(y_test.values).float().to(device), dim=1)
X_val = torch.from_numpy(X_val).float().to(device)
y_val = torch.unsqueeze(torch.from_numpy(y_val.values).float().to(device), dim=1)


# store train and val loss
train_loss_list = []
val_loss_list = []
epoch_list = []

n_features = len(housing.columns)
hidden_neurons = 150
hidden_neurons2 = 100
hidden_neurons3 = 50
# network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_features, hidden_neurons)  # hidden layer
        self.hidden2 = torch.nn.Linear(hidden_neurons, hidden_neurons2)
        self.hidden3 = torch.nn.Linear(hidden_neurons2, hidden_neurons3)
        self.predict = torch.nn.Linear(hidden_neurons3, 1)  # output layer

    def forward(self, x):
        out = F.selu(self.hidden(x)) # activation function for hidden layer
        out = F.selu(self.hidden2(out))
        out = F.selu(self.hidden3(out))
        out = self.predict(out)  # linear output
        return out

def training_loop(n_epochs, optimizer, model, loss_fn, X_train, y_train, X_val, y_val, epoch_num_of_no_improve):
    epoch_no_improve = 0
    for epoch in range(1, n_epochs+1):
        loss_train = 0.0

        outputs = model(X_train)
        loss = loss_fn(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_train += loss.item()

        epoch_list.append(epoch)
        train_loss_list.append(loss_train)

        with torch.no_grad():
            loss_val = 0.0

            outputs = model(X_val)
            loss_v = loss_fn(outputs, y_val)
            loss_val += loss_v.item()

            val_loss_list.append(loss_val)


        # set when to print info about training progress
        if epoch == 1 or epoch % 100 == 0:
            print('Epoch {}, Training loss {}, Validation loss {}'.format(epoch,
                                                                          np.sqrt(loss_train),
                                                                          np.sqrt(loss_val)))

        # early stopping

        if epoch > 1:
            if val_loss_list[-1] >= val_loss_list[-2]:
                epoch_no_improve += 1
            else:
                epoch_no_improve = 0

        if epoch_no_improve == epoch_num_of_no_improve:
            print('Early stopping:')
            print('Epoch {}, Training loss {}, Validation loss {}'.format(epoch,
                                                                          np.sqrt(loss_train),
                                                                          np.sqrt(loss_val)))
            break



def validate_on_test(model, loss_fn, X_test, y_test):
        with torch.no_grad():
            outputs = model(X_test)
            loss_test = loss_fn(outputs, y_test)

        print('Mean prediction error: {}'.format(torch.sqrt(loss_test)))

n_epochs = 10000
model = Net().to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss() # loss in report is square root of that

# epoch_num_of_no_improve = 5

training_loop(n_epochs=n_epochs,
              optimizer=optimizer,
              model=model,
              loss_fn=loss_fn,
              X_train=X_train,
              y_train=y_train,
              X_val=X_val,
              y_val=y_val,
              epoch_num_of_no_improve=25)

validate_on_test(model=model, loss_fn=loss_fn, X_test=X_test, y_test=y_test)

plt.plot(epoch_list, train_loss_list, color='blue', label='train loss')
plt.plot(epoch_list, val_loss_list, color='green', label='validation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()