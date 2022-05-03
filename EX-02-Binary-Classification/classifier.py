"""
Program to implement binary classification.
Developed by: MADITHATI YUVATEJA REDDY
RegisterNumber:  212219040069
"""
from numpy import where
from collections import Counter
from sklearn.datasets import make_blobs
from matplotlib import pyplot
X,y = make_blobs(n_samples=10, centers=2, random_state=1)
print(X.shape, y.shape)
counter = Counter(y)
print(counter)
for i in range(5):
print(X[i],y[i])
for label, _ in counter.items():
row_ix = where(y == label)[0]
pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()