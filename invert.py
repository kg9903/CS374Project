import numpy as np
import tensorflow as tf
import time
import tifffile as tiff
from data_loader import getXtrain, getYtrain, tifToVec, writeTiff
from sklearn.neural_network import MLPRegressor

Xmat_train = getXtrain()
Y_train = getYtrain()
print(Xmat_train.shape)
print(Y_train.shape)

r_c, g_c, b_c = tifToVec("validate.TIF")
Xmat_val= np.stack((r_c, g_c, b_c), axis=-1)

r_c, g_c, b_c = tifToVec("validate_inverted.TIF")
Y_val = np.stack((r_c, g_c, b_c), axis=-1)

r_c, g_c, b_c = tifToVec("test.TIF")
Xmat_test= np.stack((r_c, g_c, b_c), axis=-1)

model = MLPRegressor(hidden_layer_sizes=(32,64,32,), activation='relu',
 solver='adam', alpha=0.001, batch_size='auto', learning_rate='constant', 
 learning_rate_init=0.001, power_t=0.5, max_iter=300, shuffle=True, 
 random_state=None, tol=0.0001, verbose=True, warm_start=False, 
 momentum=0.9, nesterovs_momentum=True, early_stopping=False, 
 validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, 
 n_iter_no_change=10, max_fun=15000)

model.fit(Xmat_train, Y_train)
print("score: ", model.score(Xmat_val, Y_val))
prediction = model.predict(Xmat_test)
writeTiff(prediction)

"""
model = tf.keras.models.Sequential([
  #tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(3),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(3, activation='softmax')
  #tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(loss='mean_squared_error',
                optimizer=tf.keras.optimizers.Adam(0.001),
                metrics=['accuracy'])


print("X: ", Xmat_train.shape, " Y: ", Y_train.shape)

model.fit(Xmat_train, Y_train, epochs=50)
model.save('test_model')

model.evaluate(Xmat_val, Y_val)
prediction = model.predict(Xmat_test)
writeTiff(prediction)
"""

"""
#!/usr/bin/env python
import sys
import pandas as pd
import sklearn as sk
import scipy as sp
import tensorflow as tf
import tensorflow
import platform

import math
import random
import pandas as pd
from sklearn.neural_network import MLPRegressor

def main():

    r_c, g_c, b_c = tifToVec("original.TIF")
    Xmat_train = np.stack((r_c, g_c, b_c), axis=-1)

    r_c, g_c, b_c = tifToVec("validate.TIF")
    Xmat_val= np.stack((r_c, g_c, b_c), axis=-1)

    r_c, g_c, b_c = tifToVec("test.TIF")
    Xmat_test= np.stack((r_c, g_c, b_c), axis=-1)

    r_ci, g_ci, b_ci = tifToVec("inverted.TIF")
    Y_train = np.stack((r_ci, g_ci, b_ci), axis=-1)

    r_ci, g_ci, b_ci = tifToVec("val_inverted.TIF")
    Y_val = np.stack((r_ci, g_ci, b_ci), axis=-1)

    model = MLPRegressor(hidden_layer_sizes=(32,32), activation='relu', solver='adam', 
    alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, 
    power_t=0.5, max_iter=200, shuffle=True, random_state=1, tol=0.0001, verbose=True, 
    warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, 
    validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)

    model.fit(Xmat_train, Y_train)
    print("score: ", model.score(Xmat_val, Y_val))


if __name__ == "__main__":
    main()
"""