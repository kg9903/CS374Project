import numpy as np
import tensorflow as tf
import time
import tifffile as tiff
from data_loader import getXtrain, getYtrain, tifToVec, writeTiff, getXval, getYval, getXtest, getYtest
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
"""
Xmat_train = getXtrain()
Y_train = getYtrain()
print(Xmat_train.shape)
print(Y_train.shape)

r_c, g_c, b_c = tifToVec("validate.TIF")
Xmat_val= np.stack((r_c, g_c, b_c), axis=-1)
#Xmat_val = Xmat_val[:, 0].reshape((int(len(Xmat_val)/512),512))

r_c, g_c, b_c = tifToVec("validate_inverted.TIF")
Y_val = np.stack((r_c, g_c, b_c), axis=-1)
#Y_val = Y_val[:, 0].reshape((int(len(Y_val)/512),512))

r_c, g_c, b_c = tifToVec("test.TIF")
Xmat_test= np.stack((r_c, g_c, b_c), axis=-1) 

model =  tf.keras.models.Sequential([
  tf.keras.Input(shape=(3,)),
  tf.keras.layers.Dense(32, kernel_regularizer='l1',activation='relu'),
  tf.keras.layers.Dense(32, kernel_regularizer='l1',activation='relu'),
  tf.keras.layers.Dense(3)
  ])

model.compile(loss='mean_squared_error',
                optimizer=tf.keras.optimizers.Adam(0.001),
                metrics=['accuracy'])

model.fit(Xmat_train, Y_train, epochs= 1)
model.save("ablModel")
"""
Xmat_val = getXval()
Y_val = getYval()
Xtest = getXtest()
Ytest = getYtest()
model = tf.keras.models.load_model('ablModel')
print('====validation====')
model.evaluate(Xmat_val, Y_val)
print('++++testing++++')
model.evaluate(Xtest, Ytest)

#prediction = model.predict(Xmat_test)
#writeTiff(prediction)