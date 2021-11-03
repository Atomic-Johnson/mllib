import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from perceptron import Perceptron
from averaged_perceptron import Averaged_Perceptron as avg_p
from voted_perceptron import Voted_Perceptron as vote_p

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

print("built standard perceptron")
percy = Perceptron(notes_train, output_col, 10, r=0.4)
print("finished standard perceptron. building voted perceptron")
voted_percy = vote_p(notes_train, output_col, 10, r = 0.4)
print("finished voted perceptron. builing averaged perceptron")
avg_percy = avg_p(notes_train, output_col, 10, r=0.4)

print("Here are the weight vectors for the voted perceptron and their counts:")

for m,w,c in zip(range(voted_percy.m), voted_percy.wi, voted_percy.Cm):
    print(str(m) + " W: " + str(w) + "\t c: " + str(c))

# correct training examples
sum = 0
for i, row in notes_train.iterrows():
    if row[output_col] == voted_percy.get_label(row):
        sum = sum + 1

print("\nvoted perceptron correct training examples: {0} of {1}".format(sum, len(notes_train)))
print("voted perceptron average error on test set: {0}".format(getError(voted_percy, notes_test, output_col)))
print("voted perceptron average error on train set: {0} \n".format(getError(voted_percy, notes_train, output_col)))

print("Weight for standard perceptron: " + str(percy.w))
print("Error for standard perceptron: " + str(getError(percy, notes_test, output_col)) + "\n")

print("Weight for averaged perceptron: " + str(avg_percy.w))
print("Error for averaged perceptron: " + str(getError(avg_percy, notes_test, output_col)) + "\n")

