from pandas.core.frame import DataFrame
from pandas.core.series import Series
from ann import ann
import pandas as pd

import numpy as np
from matplotlib import pyplot as plot

def getSign(n):
    if n > 0:
        return 1
    else:
        return -1

def getError(percept, test_set: DataFrame, label_column: str):
    fail_sum = 0.0
    for i, row in test_set.iterrows():
        prediction = percept.get_prediction(row)

        fail_sum = fail_sum + abs(row[label_column] - prediction)
    
    return fail_sum / len(test_set)

output_col = "genuine"
notes_train = pd.read_csv("bank-note/train.csv", names=['variance', 'skew', 'cutosis', 'entropy', output_col])
notes_test = pd.read_csv("bank-note/test.csv", names=['variance', 'skew', 'cutosis', 'entropy', output_col])

notes_train.convert_dtypes()
notes_test.convert_dtypes()

# replace 0 with -1 in the output
notes_train[output_col] = notes_train[output_col].replace(0, -1)
notes_test[output_col] = notes_test[output_col].replace(0, -1)

fakeData = pd.DataFrame({"x1":[1,1,1], "x2":[1,1,1], "out":[1,1,1]})

nn3 = ann(fakeData,"out")

nn3.w01 = np.array([[-1,1],[-2,2],[-3,3]])
nn3.w12 = np.array([[-1,1],[-2,2],[-3,3]])
nn3.w2out = np.array([-1,2,-1.5])

test = Series({"x1":1, "x2":1, "out":1})
output = nn3.get_prediction(test)

print(output)
#That worked!!!

nn3.do_back_prop(test)

print("dL/3wij: {}".format(nn3.dldw_2out))
#That's correct!
print("dL/2wij: {}".format(nn3.dldw_12))
#This one is also correct!
print("dL/1wij: {}".format(nn3.dldw_01))
#This one matches the spreadsheet as well!


# print("evaluate convergence")
# nnbank = ann(notes_train, output_col,0,L1size=5, L2size=5)

# average_outs = []
# for i in range(100):
#     nnbank.do_SGD(i)
#     average_outs.append(nnbank.get_prediction(notes_train.loc[96]))


# plot.plot(average_outs)
# plot.show()

print("testing error rates")
widths = [5, 10, 25, 50, 100]

for width in widths:

    nnbank = ann(notes_train, output_col,60,L1size=width, L2size=width, randomW=False)
    print("test error for size {} = {}".format(width, getError(nnbank, notes_test, output_col)))
    print("train error for size {} = {}".format(width, getError(nnbank, notes_train, output_col)))



