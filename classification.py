from numpy.lib.function_base import kaiser
import pandas as pd
import numpy as np
import os
import skimage.io as io
import pywt
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
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
    def __init__ (self,method):
        self.paths = [
            'CK+48/anger',
            'CK+48/contempt',
            'CK+48/disgust',
            'CK+48/fear',
            'CK+48/happy',
            'CK+48/sadness',
            'CK+48/surprise'
        ]
        if "knn" in method:
            self.clf = make_pipeline (StandardScaler (), KNeighborsClassifier(n_neighbors=100))
        if "svm" in method:
             self.clf = make_pipeline (StandardScaler (), LinearSVC(random_state=0, tol=1e-5, max_iter=5000))
        if "tree" in method:
            self.clf=make_pipeline (StandardScaler (), DecisionTreeClassifier(random_state=0))

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
            if predicted[i] == real[i]:
                hits += 1

        return float (hits / len(predicted))


    def k_fold_cross (self, x, y, k, train_func, test_func):
        accuracy = []
        for i in range (int (len (x) / k)):
            start = i*k
            end = (i+1)*k

            x_train = x[0:start] + x[end:]
            y_train = y[0:start] + y[end:]
            x_test = x[start:end]
            y_test = y[start:end]
            
            train_func (x_train, y_train)
            y_pd = test_func (x_test)
            accuracy.append (self.calculate_accuracy (y_pd, y_test))

        return accuracy

    def random_subsample (self, x, y, k, iterations, train_func, test_func):
        accuracy = []
        for i in range (iterations):
            indexes = [np.random.randint (low=0, high=len(x)-k) for j in range (k)]
            x_train = list(x)
            y_train = list(y)
            x_test = []
            y_test = []

            for index in indexes:
                x_test.append (x[index])
                y_test.append (y[index])
                x_train.pop (index)
                y_train.pop (index)
            
            train_func (x_train, y_train)
            y_pd = test_func (x_test)
            accuracy.append (self.calculate_accuracy (y_pd, y_test))

        return accuracy