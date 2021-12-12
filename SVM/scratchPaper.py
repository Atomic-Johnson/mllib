# this is just a space for me to try some things and should be ignored

import numpy as np
from numpy.core.fromnumeric import size

alphas = np.array([1,2,3,4,5])
ys = np.array([[1,-1,1,-1,-1]])

randor = np.random.RandomState(0)
x = randor.rand(5,5)
sum = 0.0

# for i in range(5):
#     ai = alphas[0][i]
#     yi = ys[0][i]
#     xi = x[i]

#     for j in range(5):
#         sum = sum + ys[0][j] * yi * ai * alphas[0][j] * xi.T.dot(x[j])

# print(x.T.dot(x))
# print("The dumb way: " + str(sum))
print(ys * alphas)
sum = (np.dot(x, x.T) * np.dot((alphas * ys).T, (alphas*ys))).sum()
print("The otherway way: " + str(sum))
print(alphas)
print(ys)