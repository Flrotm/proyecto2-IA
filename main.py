import pandas as pd
import numpy as np
import os
import skimage.io as io
import pywt



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




path='CK+48/anger'
angerfeatures=[]
afts=[]
for image_path in os.listdir(path):
    input_path = os.path.join(path, image_path)
    img = io.imread(input_path)
    fts,_ = pywt.dwt2(img, 'haar')
    angerfeatures.append(fts)


X=flatten(angerfeatures)
print(X)
anger = np.array ([[1] for i in range (len(angerfeatures))])