import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import pickle
import tifffile as tiff

TRAIN_SET_SIZE = 42

def tifToVec(path):
    image = tiff.imread(path)
    red_channel = np.ndarray.flatten(image[:, :, 0])
    green_channel = np.ndarray.flatten(image[:, :, 1])
    blue_channel = np.ndarray.flatten(image[:, :, 2])
    #print("dimensions: ", red_channel.shape)
    return red_channel, green_channel, blue_channel

def getXtrain():
    Xtrain = np.zeros((0,3))
    for i in range(0, TRAIN_SET_SIZE):
        red_channel, green_channel, blue_channel = tifToVec('Dataset/Original_resized/'+str(i)+'.tif')
        temp = np.stack((red_channel, green_channel, blue_channel), axis=-1)
        Xtrain = np.concatenate((Xtrain, temp), axis=0)
    return Xtrain

def getYtrain():
    Ytrain = np.zeros((0,3))
    for i in range(0, TRAIN_SET_SIZE):
        red_channel, green_channel, blue_channel = tifToVec('Dataset/Inverted_resized/'+str(i)+'.tif')
        temp = np.stack((red_channel, green_channel, blue_channel), axis=-1)
        Ytrain = np.concatenate((Ytrain, temp), axis=0)
    return Ytrain

def writeTiff(prediction):
    red_r = prediction[:,0].reshape((1024,1024))
    green_r = prediction[:,1].reshape((1024,1024))
    blue_r = prediction[:,2].reshape((1024,1024))
    data = np.zeros((1024,1024,3), 'uint16')
    data[ :, :, 0] = red_r
    data[ :, :, 1] = green_r
    data[ :, :, 2] = blue_r
    tiff.imwrite('res.tif', data, photometric='rgb',tile=(32, 32),
        compression='zlib',
        predictor=True,
        metadata={'axes': 'TZCYX'},)