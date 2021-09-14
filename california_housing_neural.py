import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
# sns.set()
import tensorflow as tf
from tensorflow import keras

# DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
# HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
#     if not os.path.isdir(housing_path):
#         os.makedirs(housing_path)
#     tgz_path = os.path.join(housing_path, "housing.tgz")
#     urllib.request.urlretrieve(housing_url, tgz_path)
#     housing_tgz = tarfile.open(tgz_path)
#     housing_tgz.extractall(path=housing_path)
#     housing_tgz.close()



def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

def get_data_info():
    print(housing.info())
    data_describe = housing.describe()
    return data_describe


# fetch_housing_data()
housing = load_housing_data()
data_describe = get_data_info()

# plt.figure(1)
# housing.hist(bins=50, figsize=(20,15))
# plt.show()

housing = pd.get_dummies(housing)



# plt.figure(2)
# sns.scatterplot(data=housing, x='longitude', y='latitude', hue='median_house_value', s=1)


corr_matrix = housing.corr()

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

target = housing.pop('median_house_value')
housing['median_house_value'] = target

housing = housing.dropna()
corr_matrix_2 = housing.corr()


X = housing.loc[:, 'longitude':'population_per_household']
y =  housing['median_house_value']

X['median_income'] = np.log(X['median_income'])
X['total_rooms'] = np.log(X['total_rooms'])
X['total_bedrooms'] = np.log(X['total_bedrooms'])
X['population'] = np.log(X['population'])
X['households'] = np.log(X['households'])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

'''neural network'''

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

neural = keras.models.Sequential()
neural.add(keras.layers.Dense(300, activation='selu', input_shape=X_train.shape[1:], kernel_initializer="lecun_normal"))
neural.add(keras.layers.Dense(100, activation='selu', input_shape=X_train.shape[1:], kernel_initializer="lecun_normal"))
neural.add(keras.layers.Dense(100, activation='selu', input_shape=X_train.shape[1:], kernel_initializer="lecun_normal"))
neural.add(keras.layers.Dense(1))


neural.summary()

neural.compile(loss='mse', optimizer='nadam', metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])
history = neural.fit(X_train_scaled, y_train, epochs=150, validation_split=0.2)

score = neural.evaluate(X_test_scaled, y_test)
print('Test score:', score[1])
# print('Test accuracy:', acc)
y_fit = neural.predict(X_test_scaled)

plt.figure(5)
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.show()

neural.evaluate(X_test_scaled, y_test)


plt.show()





