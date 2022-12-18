import numpy as np
import tifffile as tiff
import tensorflow as tf
from joblib import dump, load
from data_loader import getXtrain, getYtrain, tifToVec, writeTiff, getXval, getYval
from sklearn.ensemble import HistGradientBoostingRegressor

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

Xval = getXval()
Yval = getYval()

rModel = load("red_model.joblib")
gModel = load("green_model.joblib")
bModel = load("blue_model.joblib")

print("red score: ", rModel.score(Xval[:,0].reshape(-1,1), Yval[:,0]))
print("green score: ", gModel.score(Xval[:,1].reshape(-1,1), Yval[:,1]))
print("blue score: ", bModel.score(Xval[:,2].reshape(-1,1), Yval[:,2]))

"""
model = HistGradientBoostingRegressor(loss='squared_error', quantile=None, learning_rate=0.1, 
max_iter=100, max_leaf_nodes=31, max_depth=None, min_samples_leaf=20, l2_regularization=0.0, 
max_bins=255, categorical_features=None, monotonic_cst=None, warm_start=False, early_stopping='auto', 
scoring='loss', validation_fraction=0.1, n_iter_no_change=10, tol=1e-07, verbose=1, random_state=None)

model.fit(Xmat_train[:,0].reshape(-1,1), Y_train[:,0])
dump(model, 'red_model.joblib')
print("red score: ", model.score(Xmat_val[:,0].reshape(-1,1), Y_val[:,0]))
prediction_R = model.predict(Xmat_test[:,0].reshape(-1,1))

model.fit(Xmat_train[:,1].reshape(-1,1), Y_train[:,1])
dump(model, 'green_model.joblib')
print("green score: ", model.score(Xmat_val[:,1].reshape(-1,1), Y_val[:,1]))
prediction_G = model.predict(Xmat_test[:,1].reshape(-1,1))

model.fit(Xmat_train[:,2].reshape(-1,1), Y_train[:,2])
dump(model, 'blue_model.joblib')
print("blue score: ", model.score(Xmat_val[:,2].reshape(-1,1), Y_val[:,2]))
prediction_B = model.predict(Xmat_test[:,2].reshape(-1,1))

writeTiff(np.stack((prediction_R, prediction_G, prediction_B), axis=-1))
"""
