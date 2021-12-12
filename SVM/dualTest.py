import pandas as pd
import numpy as np
from dualSVM import DualSVM

testdf = pd.read_csv("test_data.csv", header=0)
testdf.convert_dtypes()

output_col = 'y'
dsvm_100 = DualSVM(testdf, output_col, C=100/873)

testdf.insert(len(testdf.columns) - 1, "bias", 1)
y = testdf[output_col].to_numpy()
x = testdf.drop([output_col], axis=1).to_numpy()

#alphas = np.zeros(len(testdf))
alphas = np.array([1, 2, 4, 1, 16, 1.3])
sum = 0.0


for i in range(len(x)):
    ai = alphas[i]
    yi = y[i]
    xi = x[i]

    for j in range(len(x)):
        sum = sum + y[j] * yi * ai * alphas[j] * xi.T.dot(x[j])

sum = sum * 0.5 - alphas.sum()

model_out = dsvm_100.objective(alphas)

print(sum)
print(model_out)

gamma = 0.1

def gauss(xi: np.ndarray, xj: np.ndarray):
    return np.exp(- ((xi-xj)**2).sum()/gamma)


testdf = pd.read_csv("test_data.csv", header=0)
testdf.convert_dtypes()
dsvm_100_K = DualSVM(testdf, output_col, C=100/873, useGaussKernel=True, gamma=gamma)

testdf.insert(len(testdf.columns) - 1, "bias", 1)
y = testdf[output_col].to_numpy()
x = testdf.drop([output_col], axis=1).to_numpy()
sum = 0.0

for i in range(len(x)):
    ai = alphas[i]
    yi = y[i]
    xi = x[i]

    for j in range(len(x)):
        sum = sum + y[j] * yi * ai * alphas[j] * gauss(xi, x[j])

sum = sum * 0.5 - alphas.sum()

model_out = dsvm_100_K.gaussObjective(alphas)


print(sum)
print(model_out)
print(len(dsvm_100_K.alphas) - np.count_nonzero(dsvm_100_K.alphas))

