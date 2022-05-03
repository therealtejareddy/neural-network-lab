"""
Program to implement the multi class classifier.
Developed by: MADITHATI YUVATEJA REDDY
RegisterNumber: 212219040069
"""
#multi class classification 
from numpy import where
from collections import Counter
from sklearn.datasets import make_blobs 
from matplotlib import pyplot
X, y = make_blobs(n_samples=1000, centers=3, random_state=1) # summarize dataset shape
print(X.shape, y.shape)
# summarize observations by class label 
counter = Counter(y)
print(counter)
# summarize first few examples 
for i in range(10):
    print(X[i], y[i])
# plot the dataset and color the by class label 
for label, _ in counter.items():
    row_ix = where(y == label)[0]
    pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label)) 
pyplot.legend()
pyplot.show()