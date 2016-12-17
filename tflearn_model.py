# -*- coding: utf-8 -*-

"""
Geoffrey So
12/8/2016
modified sample tflearn
"""

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import glob
import cv2
import numpy as np
import pickle
import tensorflow as tf
import pandas as pd

def showimage(img):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_array(xsize, ysize, color_channels, train_list):
    X = []
    if color_channels == 1:
        colors = 0
    else:
        colors = 1
    for i in range(len(train_list)):
        if not i % 100:
            print(i)
        X.append(cv2.resize(cv2.imread(train_list[i], colors), (xsize, ysize)))
    X = np.array(X)
    X = X.reshape([-1, ysize, xsize, color_channels])
    np.save('X_%i_%i' % (xsize, color_channels), X)

class model():
    def __init__(self):
    # Data loading and preprocessing
    # Load Fisheries data
        self.data_path = r'C:\Users\gso\Documents\Fisheries'
        self.train_path = self.data_path + r'\train\train'
        self.train_list = glob.glob(self.train_path + '\*\*')
        print(self.train_list)
        self.model=None
    def load_data(self):
        reduction = 4
        xsize = int(1280/reduction)
        ysize = int(720/reduction)
        self.xsize=xsize
        self.ysize=ysize
        color_channels = 3
        self.color_channels=3
        # load X from storage
        try:
            X = np.load(self.data_path+'\\'+'X_%i_%i.npy' % (xsize, color_channels))
        except FileNotFoundError:
            # save X  so next time it can just load it into memory
            save_array(xsize, ysize, color_channels, self.train_list)
            X = np.load(self.data_path+'\\'+'X_%i_%i.npy' % (xsize, color_channels))

        Y = []
        for image_name in self.train_list:
            Y.append(image_name.split('\\')[-2])
        self.df = pd.get_dummies(Y)
        Y = self.df.values
        print(len(Y))
        n_categories = len(Y[0])
        self.n_categories=n_categories
        #X = X.reshape([-1, 28, 28, 1])
        #testX = testX.reshape([-1, 28, 28, 1])

        #shuffle data
        shuffle_index = np.arange(len(Y))
        np.random.seed(123456)
        np.random.shuffle(shuffle_index)

        X = X[shuffle_index]
        Y = Y[shuffle_index]

        split_index = int(len(Y) * 0.9)
        trainX, testX = X[:split_index], X[split_index:]
        trainY, testY = Y[:split_index], Y[split_index:]
        self.trainX=trainX
        self.testX=testX
        self.trainY=trainY
        self.testY=testY
    def simple_model(self):
        # Building convolutional network
        trainX=self.trainX
        trainY=self.trainY
        testX=self.testX
        testY=self.testY
        network = input_data(shape=[None, self.ysize, self.xsize, self.color_channels], name='input')
        network = conv_2d(network, 16, 7, strides=2, activation='relu', regularizer="L2")
        network = max_pool_2d(network, 2)
        network = local_response_normalization(network)
        network = conv_2d(network, 32, 11, strides=2, activation='relu', regularizer="L2")
        network = max_pool_2d(network, 2)
        network = local_response_normalization(network)
        network = conv_2d(network, 64, 15, strides=2, activation='relu', regularizer="L2")
        network = max_pool_2d(network, 2)
        network = local_response_normalization(network)

        network = fully_connected(network, 128, activation='tanh')
        network = dropout(network, 0.5)

        network = fully_connected(network, self.n_categories, activation='softmax')
        network = regression(network, optimizer='adam', learning_rate=1e-4,
                             loss='categorical_crossentropy', name='target')

        ### add this "fix":
        col = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for x in col:
            tf.add_to_collection(tf.GraphKeys.VARIABLES, x)

        # Training
        model = tflearn.DNN(network, tensorboard_verbose=1,tensorboard_dir="Model")
        model.fit({'input': trainX}, {'target': trainY}, n_epoch=9,
                   validation_set=({'input': testX}, {'target': testY}),
                   snapshot_step=100, show_metric=True, run_id='Fisheries',
                  batch_size=34)
        self.model=model
        model.save('test_model')
    def predict(self,outfile='submit.csv'):
        if self.model is None:
            model=tflearn.models.dnn.DNN.load('test_model.mod')
            self.model=model
        model=self.model
        # Prediction for submission
        submit_list = glob.glob(r'C:\Users\gso\Documents\Fisheries\test_stg1\test_stg1\*')
        submitX = np.array([],dtype=np.uint8)
        for i in range(len(submit_list)):
            submitX = np.append(submitX, cv2.resize(cv2.imread(submit_list[i]), (self.xsize,self.ysize)))
        submitX = submitX.reshape([-1, self.ysize, self.xsize, self.color_channels])
        submit_prediction = []
        for i in range(10):
            submit_prediction.extend(model.predict(submitX[i*100:(i+1)*100]))
        submit_prediction = np.array(submit_prediction)

        submit_filenames = []
        for i in range(len(submit_list)):
            submit_filenames.append(submit_list[i].split('\\')[-1])
        dffn = pd.DataFrame(data = submit_filenames, columns = ['image'])

        df2 = pd.DataFrame(data = submit_prediction, columns = self.df.columns)

        submit_result = pd.concat([dffn, df2], axis = 1)

        submit_result.to_csv(self.data_path+'\\'+outfile, index=False)

if __name__=='__main__':
    m=model()
    m.load_data()
    m.simple_model()
    m.predict()