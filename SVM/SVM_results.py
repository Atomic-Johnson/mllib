import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from primalSVM import PrimalSVM
from dualSVM import DualSVM

def getError(percept, test_set: DataFrame, label_column: str):
    fail_count = 0.0
    for i, row in test_set.iterrows():
        prediction = percept.get_label(row)

        if prediction != row[label_column]:
            fail_count = fail_count + 1
    
    return fail_count / len(test_set)

output_col = "genuine"
notes_train = pd.read_csv("bank-note/train.csv", names=['variance', 'skew', 'cutosis', 'entropy', output_col])
notes_test = pd.read_csv("bank-note/test.csv", names=['variance', 'skew', 'cutosis', 'entropy', output_col])

notes_train.convert_dtypes()
notes_test.convert_dtypes()

# replace 0 with -1 in the output
notes_train[output_col] = notes_train[output_col].replace(0, -1)
notes_test[output_col] = notes_test[output_col].replace(0, -1)

print("making dual SVM")


dsvm_100 = DualSVM(notes_train, output_col, C=100/873)
print("\nDual SVM C=100/873: train error = {} test error = {}".format(getError(dsvm_100, notes_train, output_col), getError(dsvm_100, notes_test, output_col)))
print("w = {}, b = {}".format(dsvm_100.w[0:-1], dsvm_100.w[len(dsvm_100.w) - 1]))

dsvm_500 = DualSVM(notes_train, output_col, C=500/873)
print("\nDual SVM C=500/873: train error = {} test error = {}".format(getError(dsvm_500, notes_train, output_col), getError(dsvm_500, notes_test, output_col)))
print("w = {}, b = {}".format(dsvm_500.w[0:-1], dsvm_500.w[len(dsvm_500.w) - 1]))

dsvm_700 = DualSVM(notes_train, output_col, C=700/873)
print("\nDual SVM C=700/873: train error = {} test error = {}".format(getError(dsvm_700, notes_train, output_col), getError(dsvm_700, notes_test, output_col)))
print("w = {}, b = {}".format(dsvm_700.w[0:-1], dsvm_700.w[len(dsvm_700.w) - 1]))

print("\nbuilding primal SVM")

print("Primal SVM C=100/873: w =", end=" ")
svmprime_100 = PrimalSVM(notes_train, output_col, 100, C=100/873)
print("Primal SVM C=100/873: train error = {} test error = {}".format(getError(svmprime_100, notes_train, output_col), getError(svmprime_100, notes_test, output_col)))

print("Primal SVM C=500/873: w =", end=" ")
svmprime_500 = PrimalSVM(notes_train, output_col, 100, C=500/873)
print("Primal SVM C=500/873: train error = {} test error = {}".format(getError(svmprime_500, notes_train, output_col), getError(svmprime_500, notes_test, output_col)))

print("Primal SVM C=700/873: w =", end=" ")
svmprime_700 = PrimalSVM(notes_train, output_col, 100, C=700/873)
print("Primal SVM C=700/873: train error = {} test error = {}".format(getError(svmprime_700, notes_train, output_col), getError(svmprime_700, notes_test, output_col)))


print("\nrunning SVM with the margin function specified in part b")
print("Primal SVM 2.2.b C=100/873: w =", end=" ")
svmprime_100 = PrimalSVM(notes_train, output_col, 100, C=100/873, gammaFunc='B')
print("Primal SVM C=100/873 with gamma func for partb: train error = {} test error = {}".format(getError(svmprime_100, notes_train, output_col), getError(svmprime_100, notes_test, output_col)))

print("Primal SVM 2.2.b C=500/873: w =", end=" ")
svmprime_500 = PrimalSVM(notes_train, output_col, 100, C=500/873, gammaFunc='B')
print("Primal SVM C=500/873 with gamma func for partb: train error = {} test error = {}".format(getError(svmprime_500, notes_train, output_col), getError(svmprime_500, notes_test, output_col)))

print("Primal SVM 2.2.b C=700/873: w =", end=" ")
svmprime_700 = PrimalSVM(notes_train, output_col, 100, C=700/873, gammaFunc='B')
print("Primal SVM C=700/873 with gamma func for partb: train error = {} test error = {}".format(getError(svmprime_700, notes_train, output_col), getError(svmprime_700, notes_test, output_col)))
