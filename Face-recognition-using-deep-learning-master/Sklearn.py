import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy
 
a = numpy.array([])
 
newArray = numpy.append (a, [10, 11, 12])

print(newArray)

X = np.array([[0.1,0.1],
             [0.2,0.2],
             [0.3,0.3],
             [15,15],
             [10,10],
             [11,11]])
y = ["wanne","wanne","wanne","henk","jasper","jasper"]
clf = svm.SVC(kernel='linear', C = 1, probability=True, gamma=0.001)
svc = clf.fit(X,y)
y = clf.predict_proba([[0.14,0.14]])

it = np.nditer(newArray, flags=['f_index'])
while not it.finished:
    if it[0]>10:
        print("hello")
    it.iternext()

g = clf.predict([[0.14,14]])
print(g[0])
print(clf.decision_function([[0.14,14]]))