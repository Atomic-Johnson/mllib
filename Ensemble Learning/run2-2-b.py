import sys
import os
from typing import List
import pandas as df
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
from treeBaggerMultiProc import BaggedTree

# for arguments this expects bagRunner.py <startSize> <endSize> <outputFile.csv>
# example: $ python bagRunner.py 1 350 bag1.csv
def getErrors(dt, test_set: DataFrame, label_column: str, T:int):
    fail_counts = np.zeros(T)
    for j, row in test_set.iterrows():
        row_dict = row.to_dict()
        predictions:List = dt.getLabelsUpToT(row_dict,T)

        for i in range(0,T):
            if predictions[i] != row_dict[label_column]:
                fail_counts[i] = fail_counts[i] + 1
    
    return fail_counts / len(test_set)

if __name__ == '__main__':

    if len(sys.argv) == 4:
        startSize = int(sys.argv[1])
        endSize = int(sys.argv[2])
        outFile = sys.argv[3]
    else:
        startSize = 1
        endSize = 500
        outFile = "fullBagOutput.csv"

    print(os.getcwd())

    bank_frame_train = df.read_csv("bank/train.csv", names=["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", \
            "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"])

    bank_frame_test = df.read_csv("bank/test.csv", names=["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", \
        "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"])

    bank_frame_train["y"] = bank_frame_train["y"].replace("no", -1)
    bank_frame_train["y"] = bank_frame_train["y"].replace("yes", 1)
    bank_frame_test["y"] = bank_frame_train["y"].replace("no", -1)
    bank_frame_test["y"] = bank_frame_train["y"].replace("yes", 1)

    # convert to numeric types
    bank_frame_test.convert_dtypes()
    bank_frame_train.convert_dtypes()

    bank_schema = \
        { "age":[], \
            "job":["admin.","unknown","unemployed","management","housemaid","entrepreneur","student", "blue-collar","self-employed","retired","technician","services" ] ,\
            "marital":[ "married","divorced","single"] ,\
            "education":[ "unknown","secondary","primary","tertiary"] ,\
            "default":[  "yes","no"] ,\
            "balance":[ ] ,\
            "housing":[ "yes","no"] ,\
            "loan":[ "yes","no"] ,\
            "contact":[  "unknown","telephone","cellular"] ,\
            "day":[ ] ,\
            "month":[ "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"] ,\
            "duration":[ ] ,\
            "campaign":[ ] ,\
            "pdays":[ ] ,\
            "previous":[ ] ,\
            "poutcome":["unknown","other","failure","success"], \
            "y": ["yes", "no"]}

    iteration_results = df.DataFrame(columns=["iteration", "train error", "test error"],dtype=float)


    print("Training the tree bagging")
    bag_of_trees = BaggedTree(bank_frame_train, endSize, 'y', bank_schema)
    print("Finished training. getting error info")

    np_train_errors = getErrors(bag_of_trees, bank_frame_train, 'y', endSize)
    np_test_errors = getErrors(bag_of_trees, bank_frame_test, 'y', endSize)

    for i in range(0, endSize):
        print(i, end=" ", flush=True)
        iteration_results = iteration_results.append({"iteration": i+1, \
            "train error":np_train_errors[i], "test error":np_test_errors[i] }, ignore_index=True)

    plt.plot(range(1,endSize+1), np_train_errors, label="train error")
    plt.plot(range(1,endSize+1), np_test_errors, label="test")
    plt.title("bagged tree errors")
    plt.legend(["train error", "test error"])
    plt.show()
    iteration_results.to_csv(outFile)


    print("finished run " + str(startSize) + " " + str(endSize) + " " + outFile)