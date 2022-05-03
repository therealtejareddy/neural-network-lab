"""
Program to implement random classification.
Developed by   : MADITHATI YUVATEJA REDDY
RegisterNumber :  212219040069
"""

import matplotlib.pyplot as plt
from sklearn import datasets
X,y = datasets.make_blobs(n_samples=100,n_features=2,centers=2,cluster_std=1.05,random_state=2)
fig = plt.figure(figsize=(10,8))
plt.plot(X[:, 0][y==0], X[:, 1][y==0], 'r^')
plt.plot(X[:, 0][y==1], X[:, 1][y==1], 'bs')
plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.title("Random Classification Data with 2 classes")