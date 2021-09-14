from sklearn.datasets import load_diabetes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
sns.set()

'''load data'''
X, y = load_diabetes(return_X_y= True)

X = pd.DataFrame(X)
# print(X.isnull().sum().sum()) #nie ma żadnych NaN
columns_names = X.columns.values
print(X.columns.values.tolist())
diabetes = X
diabetes['target'] = y

'''look for correlation'''
plt.figure(0)
# sns.pairplot(diabetes)
sns.pairplot(diabetes,    x_vars=columns_names,    y_vars='target')


correlations = diabetes.corr()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=42)



def evaluate_model(estimator):
    model = make_pipeline(StandardScaler(), estimator)
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_train)
    print('\nAccuracy on train set:', model.score(X_train, y_train))
    # mse = np.sqrt(mean_squared_error(y_train, y_predicted))
    # print('Mean squared error:', mse)

    print('Accuracy on val set:', model.score(X_test, y_test))

estimators = [LinearRegression(), RandomForestRegressor(max_depth=6)]
for i in estimators:
    evaluate_model(i)

plt.show()