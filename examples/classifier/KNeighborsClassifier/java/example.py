from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

from onl.nok.sklearn.Porter import port

X, y = load_iris(return_X_y=True)
clf = KNeighborsClassifier()
clf.fit(X, y)

print(clf)

# Cheese!

# print(port(clf))
