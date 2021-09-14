from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation='selu', input_shape=X_train.shape[1:]),
    keras.layers.Dense(1)
])

model.compile(loss="mse", optimizer="sgd", metrics='mean_squared_error')
history = model.fit(X_train_scaled, y_train, epochs=100, validation_data=(X_valid_scaled, y_valid))
score, acc = model.evaluate(X_test_scaled, y_test)
print('Test score:', score)
print('Test accuracy:', acc)
y_fit = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_fit)
print('Typical prediction error', np.sqrt(mse))


plt.figure()
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()


