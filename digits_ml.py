from sklearn.datasets import load_digits
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import mode
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

sns.set()

'''load data'''
digits = load_digits()
X = digits.data
y = digits.target
print(X.shape)
print(digits.target.shape)

'''2D visualise'''
pca = PCA(n_components=2, random_state=42)
pca_projected = pca.fit_transform(digits.data)
pca_projected = pd.DataFrame(pca_projected)

hue = digits.target

plt.figure(1)
sns.scatterplot(data=pca_projected, x=0, y=1, hue=hue, palette='hls')


'''check for number of components'''
pca = PCA().fit(X)
plt.figure(2)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

'''train, test and val sets'''
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=42)


'''preprocessing'''
scaler = StandardScaler()
pca = PCA()
# pca_projected = pca.fit_transform(digits.data)
Xtrain = scaler.fit_transform(Xtrain)
Xtest = scaler.transform(Xtest)



# '''support vector machines'''
# print('\nSupport vector machines model')
#
# svc = SVC()
# param_grid = {'svc__C': [0.001, 0.01, 1, ], 'svc__gamma': [0.001, 0.01, 1],
#               'svc__kernel':['linear', 'poly', 'rbf', 'sigmoid'], 'pca__n_components':[1,18]}
#
# svc_model = Pipeline(steps=[('pca', pca), ('svc', svc)])
# grid_search_svc = GridSearchCV(svc_model, param_grid, cv=5, n_jobs=-1)



# grid_search_svc.fit(Xtrain, ytrain)
# print("Best parameters: {}".format(grid_search_svc.best_params_))
# print("Best cross-validation score: {:.2f}".format(grid_search_svc.best_score_))
#
# print('Train accuracy', grid_search_svc.score(Xtrain, ytrain))
# print('Test accuracy', grid_search_svc.score(Xtest, ytest))
# yfit = grid_search_svc.predict(Xtest)
# clusters = grid_search_svc.predict(digits.data)
#
# '''make labels'''
# labels = np.zeros_like(clusters)
# for i in range(10):
#     mask = (clusters == i)
#     labels[mask] = mode(digits.target[mask])[0]
#
#
# '''Report'''
# # print(classification_report(ytest, yfit))
#
# plt.figure(3)
# mat = confusion_matrix(ytest, yfit)
# sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
# plt.xlabel('true label')
# plt.ylabel('predicted label')

# '''Random Forest'''
# print('\nRandom Forest model')
# forest = RandomForestClassifier(max_depth=7)
# forest_model = make_pipeline(pca, forest)
#
# forest_model.fit(Xtrain, ytrain)
# print('Train accuracy', forest_model.score(Xtrain, ytrain))
# print('Test accuracy', forest_model.score(Xtest, ytest))
# yfit = forest_model.predict(Xtest)
# clusters = forest_model.predict(digits.data)
#
# '''make labels'''
# labels = np.zeros_like(clusters)
# for i in range(10):
#     mask = (clusters == i)
#     labels[mask] = mode(digits.target[mask])[0]
#
#
# '''Report'''
# # print(classification_report(ytest, yfit))
#
# plt.figure(4)
# mat = confusion_matrix(ytest, yfit)
# sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
# plt.xlabel('true label')
# plt.ylabel('predicted label')
#

# '''Logistic Regression'''
# print('\nLogistic Regression model')
# log_reg = LogisticRegression(solver='lbfgs', max_iter=100000)
# log_reg_model = make_pipeline(pca, log_reg)
#

# log_reg_model.fit(Xtrain, ytrain)
# print('Train accuracy', log_reg_model.score(Xtrain, ytrain))
# print('Test accuracy', log_reg_model.score(Xtest, ytest))
# yfit = log_reg_model.predict(Xtest)
# clusters = log_reg_model.predict(digits.data)
#
#
# '''Report'''
# # print(classification_report(ytest, yfit))
#
# plt.figure(5)
# mat = confusion_matrix(ytest, yfit)
# sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
# plt.xlabel('true label')
# plt.ylabel('predicted label')
#
# '''Voting'''
# print('\nVoting model')
# voting_clf = VotingClassifier(
#     estimators=[('svc', svc),('forest', forest),('log_reg', log_reg)],
#     voting='hard')
# voting_model = make_pipeline(pca, voting_clf)
#
# voting_model.fit(Xtrain, ytrain)
# print('Train accuracy', voting_model.score(Xtrain, ytrain))
# print('Test accuracy', voting_model.score(Xtest, ytest))
# yfit = voting_model.predict(Xtest)
# clusters = voting_model.predict(digits.data)
#
# '''Report'''
# # print(classification_report(ytest, yfit))
#
# plt.figure(6)
# mat = confusion_matrix(ytest, yfit)
# sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
# plt.xlabel('true label')
# plt.ylabel('predicted label')


plt.show()
