from sklearn.utils import shuffle

from sklearn.neural_network import MLPClassifier

import numpy as np

import pickle

import gzip

import os

# import numpy as np;

import cv2 as cv;

def gatherDataset():

    print('gathering dataset')
    datasetX = []
    datasetY = []

    for dir in os.listdir('./datasets/'):
        if os.path.isdir('./datasets/' + dir):
            for fle in os.listdir('./datasets/' + dir + '/'):
                datasetX.append(cv.imread('./datasets/' + dir + '/' + fle))
                datasetY.append(dir)

    return datasetX, datasetY

def trainModel(X, y):

    print('flattening images')
    flat_X = []
    for x in X:
        flat_X.append(x.flatten())

    print('training model')
    nn = MLPClassifier()
    nn.fit(flat_X, y)
    return nn

def testModel(X, y, model):
    print('testing model')
    X_flat_test = [x.copy().flatten() for x in X]
    acc = model.score(X_flat_test, y)
    print('Testing accuracy: ' + str(acc))

def main():

    X, y = gatherDataset()
    X, y = shuffle(X, y, random_state=0)
    
    lengthDataset = len(X)
    trainAmount = int(lengthDataset*.7)

    X_train = X[0:trainAmount]
    y_train = y[0:trainAmount]
    X_test = X[trainAmount + 1:]
    y_test = y[trainAmount + 1:]

    model = trainModel(X_train, y_train)
    pickle.dump(model, gzip.open('trained_model.pklz', 'wb'))

    testModel(X_test, y_test, model)

    



if __name__ == '__main__':
    main();