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
    def __init__ (self, training_method=0, k=4):
        self.paths = [
            'CK+48/anger',
            'CK+48/contempt',
            'CK+48/disgust',
            'CK+48/fear',
            'CK+48/happy',
            'CK+48/sadness',
            'CK+48/surprise'
        ]
        self.clf = make_pipeline (StandardScaler (), LinearSVC(random_state=0, tol=1e-5))
        self.x = []
        self.y = []
        self.training_method = training_method
        self.k = k
    
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

    def wavelet(pictures,n):
        ca=pictures
        for i in range(n):
            ca,_=pywt.dwt2(ca, 'haar')
        return ca

    def flatten(matrixs):
        vectors=[]
        for i in range(len(matrixs)):
            vectors = vectors + [matrixs[i].reshape(24*24).tolist()]
        return vectors

    def fit (self, x, y):
        self.clf.fit (x, y)

    def classify (self, x):
        return self.clf.predict (x)