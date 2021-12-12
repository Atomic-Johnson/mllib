import numpy as np
import sys
import pandas as pd
from pandas.core.frame import DataFrame
from dualSVM import DualSVM

def getError(percept, test_set: DataFrame, label_column: str):
    fail_count = 0.0
    for i, row in test_set.iterrows():
        prediction = percept.get_label(row)

        if prediction != row[label_column]:
            fail_count = fail_count + 1
    
    return fail_count / len(test_set)

if len(sys.argv) < 2:
    print("ERROR: must have one command line argument for gamma")
    gamma = 0.1
else:
    gamma = float(sys.argv[1])

output_col = "genuine"
notes_train = pd.read_csv("bank-note/train.csv", names=['variance', 'skew', 'cutosis', 'entropy', output_col])
notes_test = pd.read_csv("bank-note/test.csv", names=['variance', 'skew', 'cutosis', 'entropy', output_col])

notes_train.convert_dtypes()
notes_test.convert_dtypes()

# replace 0 with -1 in the output
notes_train[output_col] = notes_train[output_col].replace(0, -1)
notes_test[output_col] = notes_test[output_col].replace(0, -1)

dsvm_100 = DualSVM(notes_train, output_col, C=100/873, useGaussKernel=True, gamma=gamma)
print("\nDual SVM C=100/873 gamma={}: train error = {} test error = {}".format(gamma, getError(dsvm_100, notes_train, output_col), getError(dsvm_100, notes_test, output_col)))
print("Dual SVM C=100/873 gamma={}: #SVs = {} ".format(gamma,  np.count_nonzero(dsvm_100.alphas)))

dsvm_500 = DualSVM(notes_train, output_col, C=500/873, useGaussKernel=True, gamma=gamma)
print("\nDual SVM C=500/873 gamma={}: train error = {} test error = {}".format(gamma, getError(dsvm_500, notes_train, output_col), getError(dsvm_500, notes_test, output_col)))
print("Dual SVM C=500/873 gamma={}: #SVs = {} ".format(gamma, np.count_nonzero(dsvm_500.alphas)))

np.savetxt(str(gamma) + ".csv", dsvm_500.alphas, header=str(gamma), delimiter=",", fmt='%s')

dsvm_700 = DualSVM(notes_train, output_col, C=700/873, useGaussKernel=True, gamma=gamma)
print("\nDual SVM C=700/873 gamma={}: train error = {} test error = {}".format(gamma, getError(dsvm_700, notes_train, output_col), getError(dsvm_700, notes_test, output_col)))
print("Dual SVM C=700/873 gamma={}: #SVs = {} ".format(gamma, np.count_nonzero(dsvm_700.alphas)))

