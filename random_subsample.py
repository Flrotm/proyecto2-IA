from classification import Classfication

clf = Classfication ()

characteristics, emotions = clf.generate_datasets ()

k = 40
iterations = 10

accuracy = clf.random_subsample (characteristics, emotions, k, iterations, clf.clf.fit, clf.clf.predict)
print (accuracy)
