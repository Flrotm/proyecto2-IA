from classification import Classfication
#method="svm"
#method="knn"
method="dtree"
clf = Classfication (method)

characteristics, emotions = clf.generate_datasets ()

k = int (len (characteristics) / 10)

accuracy = clf.k_fold_cross (characteristics, emotions, k, clf.clf.fit, clf.clf.predict)
print (accuracy)