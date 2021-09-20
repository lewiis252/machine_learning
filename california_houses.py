import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNet, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.metrics import mean_squared_error, r2_score
# sns.set()

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






# '''check for number of components'''
# pca = PCA().fit(X)
# plt.figure(4)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance')
# plt.show()

'''Linear Regression'''

print('\nLinear Regression model')

param_grid = {'pca__n_components': [4,5,6,7]}
pipe = Pipeline(steps=[('scaler', StandardScaler()),('pca', PCA(random_state=42)), ('lin_reg', LinearRegression())])

model = RandomizedSearchCV(pipe, param_grid, cv=5, n_jobs=-1, random_state=42, )
model.fit(X_train, y_train)

print("Best parameters: {}".format(model.best_params_))
print("Best cross-validation score: {:.2f}".format(model.best_score_))

print('Train accuracy', model.score(X_train, y_train))
print('Test accuracy', model.score(X_test, y_test))
y_fit = model.predict(X_test)
mse = mean_squared_error(y_test, y_fit)
print('Typical prediction error', np.sqrt(mse))
# cvres = model.cv_results_
# scores = np.sqrt(-(cvres["mean_test_score"]))
# print(np.mean(scores))
# print(np.std(scores))


# '''Random Forest Regression'''
#
# print('\nRandom Forest Regression model')
#
# param_grid = {'forest__max_depth':[20,21,22,23]}
# pipe = Pipeline(steps=[('forest', RandomForestRegressor(random_state=42))])
#
# model = RandomizedSearchCV(pipe, param_grid, cv=5, n_jobs=-1, random_state=42)
# model.fit(X_train, y_train)
#
# print("Best parameters: {}".format(model.best_params_))
# print("Best cross-validation score: {:.2f}".format(model.best_score_))
#
# print('Train accuracy', model.score(X_train, y_train))
# print('Test accuracy', model.score(X_test, y_test))
# y_fit = model.predict(X_test)
# mse = mean_squared_error(y_test, y_fit)
# print('Typical prediction error', np.sqrt(mse))
# # cvres = model.cv_results_
# # scores = np.sqrt(-(cvres["mean_test_score"]))
# # print(np.mean(scores))
# # print(np.std(scores))
#
#
#
#
#
#
# '''Elastic Net Regression'''
#
# print('\nElastic Net Regression model')
#
# param_grid = {'pca__n_components': [4,5,6], 'ela_net__alpha':[0.1, 1, 10, 100,200],
#               'ela_net__l1_ratio':[0, 0.5, 1]}
# pipe = Pipeline(steps=[('scaler', StandardScaler()),('pca', PCA(random_state=42)),
#                        ('ela_net', ElasticNet(max_iter=1000000, random_state=42))])
#
# model = RandomizedSearchCV(pipe, param_grid, cv=5, n_jobs=-1, random_state=42)
# model.fit(X_train, y_train)
#
# print("Best parameters: {}".format(model.best_params_))
# print("Best cross-validation score: {:.2f}".format(model.best_score_))
#
# print('Train accuracy', model.score(X_train, y_train))
# print('Test accuracy', model.score(X_test, y_test))
# y_fit = model.predict(X_test)
# mse = mean_squared_error(y_test, y_fit)
# print('Typical prediction error', np.sqrt(mse))
# # cvres = model.cv_results_
# # scores = np.sqrt(-(cvres["mean_test_score"]))
# # print(np.mean(scores))
# # print(np.std(scores))
#
#
# '''SGD Regression'''
# print('\nSGD Regression model')
#
# param_grid = {'pca__n_components': [4,5,6], 'sgd__alpha':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100], 'sgd__l1_ratio':[0, 0.5, 1]}
# pipe = Pipeline(steps=[('scaler', StandardScaler()),('pca', PCA(random_state=42)), ('sgd', SGDRegressor(max_iter=10000, random_state=42))])
#
# model = RandomizedSearchCV(pipe, param_grid, cv=5, n_jobs=-1, random_state=42)
# model.fit(X_train, y_train)
#
# print("Best parameters: {}".format(model.best_params_))
# print("Best cross-validation score: {:.2f}".format(model.best_score_))
#
# print('Train accuracy', model.score(X_train, y_train))
# print('Test accuracy', model.score(X_test, y_test))
# y_fit = model.predict(X_test)
# mse = mean_squared_error(y_test, y_fit)
# print('Typical prediction error', np.sqrt(mse))
# # cvres = model.cv_results_
# # scores = np.sqrt(-(cvres["mean_test_score"]))
# # print(np.mean(scores))
# # print(np.std(scores))
#
# '''Linear SVR Regression'''
# print('\nLinear SVR Regression model')
#
# param_grid = {'pca__n_components': [4,5,6], 'lsvr__C':[1, 10, 100,200,300,400,500,600]}
# pipe = Pipeline(steps=[('scaler', StandardScaler()),('pca', PCA(random_state=42)), ('lsvr', LinearSVR(max_iter=10000, random_state=42))])
#
# model = RandomizedSearchCV(pipe, param_grid, cv=5, n_jobs=-1, random_state=42)
# model.fit(X_train, y_train)
#
# print("Best parameters: {}".format(model.best_params_))
# print("Best cross-validation score: {:.2f}".format(model.best_score_))
#
# print('Train accuracy', model.score(X_train, y_train))
# print('Test accuracy', model.score(X_test, y_test))
# y_fit = model.predict(X_test)
# mse = mean_squared_error(y_test, y_fit)
# print('Typical prediction error', np.sqrt(mse))
# # cvres = model.cv_results_
# # scores = np.sqrt(-(cvres["mean_test_score"]))
# # print(np.mean(scores))
# # print(np.std(scores))
#
# # nonlinear svr take a lot of time and perform poorly, dunno
# # '''Nonlinear SVR Regression'''
# # print('\nSVR Regression model')
# #
# # param_grid = {'pca__n_components': [4,5,6], 'svr__C':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
# #               'svr__kernel':['poly', 'rbf', 'sigmoid'], 'svr__degree':[2,3,4,5]}
# # pipe = Pipeline(steps=[('scaler', StandardScaler()),('pca', PCA(random_state=42)), ('svr', SVR(max_iter=10000))])
# #
# # model = RandomizedSearchCV(pipe, param_grid, cv=5, n_jobs=-1,random_state=42)
# # model.fit(X_train, y_train)
# #
# # print("Best parameters: {}".format(model.best_params_))
# # print("Best cross-validation score: {:.2f}".format(model.best_score_))
# #
# # print('Train accuracy', model.score(X_train, y_train))
# # print('Test accuracy', model.score(X_test, y_test))
# # yfit = model.predict(X_test)
#
'''KNeighbours Regression'''
print('\nKNeigbours Regression model')

param_grid = {'pca__n_components': [4,5,6,7,8,9,10], 'kn__n_neighbors':np.arange(4,14),
              'kn__p':[1,2], 'kn__algorithm':['ball_tree', 'kd_tree', 'brute']}
pipe = Pipeline(steps=[('scaler', StandardScaler()),('pca', PCA(random_state=42)), ('kn', KNeighborsRegressor())])

model = RandomizedSearchCV(pipe, param_grid, cv=5, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

print("Best parameters: {}".format(model.best_params_))
print("Best cross-validation score: {:.2f}".format(model.best_score_))

print('Train accuracy', model.score(X_train, y_train))
print('Test accuracy', model.score(X_test, y_test))

y_fit = model.predict(X_test)

mse = mean_squared_error(y_test, y_fit)
print('Typical prediction error', np.sqrt(mse))
# cvres = model.cv_results_
# scores = np.sqrt(-(cvres["mean_test_score"]))
# print(np.mean(scores))
# print(np.std(scores))
#
# '''neural network'''
# from sklearn.neural_network import MLPRegressor
#
# print('\n MLP Regression model')
# scaler = StandardScaler()
#
#
# param_grid = {'pca__n_components': [5,6,7], 'mlp__activation':['relu'],
#               'mlp__solver':['lbfgs'], 'mlp__alpha':[0.001, 0.01, 0.1]}
# pipe = Pipeline(steps=[('scaler', StandardScaler()),('pca', PCA(random_state=42)), ('mlp', MLPRegressor(random_state=42, max_iter=800, hidden_layer_sizes = (50, 50)))])
#
# model = RandomizedSearchCV(pipe, param_grid, cv=5, n_jobs=-1,random_state=42)
# model.fit(X_train, y_train)
#
# print("Best parameters: {}".format(model.best_params_))
# print("Best cross-validation score: {:.2f}".format(model.best_score_))
#
# print('Train accuracy', model.score(X_train, y_train))
# print('Test accuracy', model.score(X_test, y_test))
# y_fit = model.predict(X_test)
# mse = mean_squared_error(y_test, y_fit)
# print('Typical prediction error', np.sqrt(mse))
#
# plt.show()
