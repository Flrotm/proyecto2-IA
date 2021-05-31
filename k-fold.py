from classification import Classfication
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
ites=10
#method="svm"
#method="knn"
method="dtree"
clftree = Classfication (method)
characteristics, emotions = clftree.generate_datasets ()

k = int (len (characteristics) / 10)
accuracy1 = clftree.k_fold_cross (characteristics, emotions, k,clftree.clf.fit, clftree.clf.predict)
print (accuracy1)
print("-----------------------------------------------------------")

method="knn"
clfknn = Classfication (method)
characteristics, emotions = clfknn.generate_datasets ()
k = int (len (characteristics) / 10)
accuracy2 = clfknn.k_fold_cross (characteristics, emotions,k,clfknn.clf.fit, clfknn.clf.predict)
print (accuracy2)
print("-----------------------------------------------------------")

method="svm"
#method="knn"
#method="dtree"
clfsvm = Classfication (method)

characteristics, emotions = clfsvm.generate_datasets ()

k = int (len (characteristics) / 10)

accuracy3 = clfsvm.k_fold_cross (characteristics, emotions, k, clfsvm.clf.fit, clfsvm.clf.predict)
print (accuracy3)
def avg(lst):
    return sum(lst) / len(lst)
D={
    "knn":1-avg(accuracy2),
    "svm":1-avg(accuracy1),
    "dtree":1-avg(accuracy3)
}
def V(accuracy):
    ans=0
    meanerror=1-avg(accuracy)
    for i in accuracy:
        ans=ans+(((1-i)-meanerror))**2
    return ans/len(accuracy)


V={
    "knn":V(accuracy2),
    "svm":V(accuracy1),
    "dtree":V(accuracy3)
}
print(V)
plt.bar(range(len(D)), list(D.values()), align='center')
plt.xticks(range(len(D)), list(D.keys()))
plt.ylabel="Error"
plt.show()
