from numpy.lib.function_base import kaiser
import pandas as pd
import numpy as np
import os
import skimage.io as io
import pywt
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from enum import Enum


class Emotion(Enum):
    ANGER = 0
    CONTEMPT = 1
    DISGUST = 2
    FEAR = 3
    HAPPY = 4
    SADNESS = 5
    SURPRISE = 6


class Classfication:
    
    # training_method: 0 = k-fold
    def __init__ (self):
        self.paths = [
            'CK+48/anger',
            'CK+48/contempt',
            'CK+48/disgust',
            'CK+48/fear',
            'CK+48/happy',
            'CK+48/sadness',
            'CK+48/surprise'
        ]
        self.clf = make_pipeline (StandardScaler (), LinearSVC(random_state=0, tol=1e-5, max_iter=5000))
        self.x = []
        self.y = []
    
    def generate_datasets (self):
        self.x = []
        self.y = []
        i = 0
        for path in self.paths:
            features = []
            for image_path in os.listdir(path):
                input_path = os.path.join(path, image_path)
                img = io.imread(input_path)
                fts,_ = pywt.dwt2(img, 'haar')
                features.append(fts)
            features = self.flatten (features)
            self.x += features
            self.y += [i for f in features]
            i += 1
        return self.x, self.y

    def wavelet(self, pictures,n):
        ca=pictures
        for i in range(n):
            ca,_=pywt.dwt2(ca, 'haar')
        return ca

    def flatten(self, matrixs):
        vectors=[]
        for i in range(len(matrixs)):
            vectors = vectors + [matrixs[i].reshape(24*24).tolist()]
        return vectors

    def fit (self, x, y):
        self.clf.fit (x, y)

    def classify (self, x):
        return self.clf.predict (x)

    def calculate_accuracy (self, predicted, real):
        hits = 0
        for i in range (len(predicted)):
            print (predicted[i], real[i])
            if predicted[i] == real[i]:
                hits += 1

        print (hits)
        return float (hits / len(predicted))


    def k_fold_cross (self, x, y, k):
        accuracy = []
        for i in range (int (len (x) / k)):
            start = i*k
            end = (i+1)*k

            x_train = x[0:start] + x[end:]
            y_train = y[0:start] + y[end:]
            x_test = x[start:end]
            y_test = y[start:end]
            self.clf.fit (x_train, y_train)
            y_pd = self.clf.predict (x_test)
            accuracy.append (self.calculate_accuracy (y_pd, y_test))
        return accuracy