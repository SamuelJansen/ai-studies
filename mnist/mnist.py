import time
import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt


import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf


from python_helper import Constant as c
from python_helper import RandomHelper, StringHelper, ObjectHelper, log


import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


originalDataset = fetch_openml('mnist_784', cache=False)
X = originalDataset.data.astype('float32')
Y = originalDataset.target.astype('int64')
print(f'X type: {type(X)}')
print(f'Y type: {type(Y)}')


encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = np_utils.to_categorical(encoded_Y)


def baseline_model():
	model = Sequential()
	# model.add(Dense(40, input_dim=4, activation='relu'))
	model.add(Dense(40, activation='sigmoid'))
	model.add(Dense(10, activation='sigmoid'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=10, verbose=0)


kfold = KFold(n_splits=10, shuffle=True)


results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))



#############################################################
# originalDataset = fetch_openml('mnist_784', cache=False)
# x = originalDataset.data.astype('float32')
# y = originalDataset.target.astype('int64')
# print(f'x type: {type(x)}')
# print(f'y type: {type(y)}')


# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# print(f'x_train type: {type(x_train)}')
# print(f'y_train type: {type(y_train)}')
# print(f'x_test type: {type(x_test)}')
# print(f'y_test type: {type(y_test)}')
#
#
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Dense(784, input_shape=x_train.shape, activation='sigmoid'))
# model.add(tf.keras.layers.Dense(784, activation='sigmoid'))
# model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
#
#
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
#
# model.fit(x_train, y_train, epochs=100)
#
#
# model.evaluate(x_test, y_test)
