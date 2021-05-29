from classification import Classfication

clf = Classfication ()

characteristics, emotions = clf.generate_datasets ()

k = int (len (characteristics) / 9)

accuracy = clf.k_fold_cross (characteristics, emotions, k)
print (accuracy)