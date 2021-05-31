from classification import Classfication

clf = Classfication ()

characteristics, emotions = clf.generate_datasets ()

k = int (len (characteristics) / 10)

clf.plot_cMatrix (characteristics, emotions, k, clf.clf.fit, clf.clf.predict)
