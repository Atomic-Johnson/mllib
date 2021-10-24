from randomForest import RandomForest
from treeBaggerMultiProc import BaggedTree
from decisionTree import DecisionTree
import pandas as df
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pandas.core.frame import DataFrame
from typing import List

def getError(dt, test_set: DataFrame, label_column: str):
    failures = 0
    for j, row in test_set.iterrows():
        row_dict = row.to_dict()
        if row[label_column] != dt.getLabel(row_dict):
            failures = failures + 1
    return failures/ len(test_set)

def getErrors(dt, test_set: DataFrame, label_column: str, T:int):
    fail_counts = np.zeros(T)
    for j, row in test_set.iterrows():
        row_dict = row.to_dict()
        predictions:List = dt.getLabelsUpToT(row_dict,T)

        for i in range(0,T):
            if predictions[i] != row_dict[label_column]:
                fail_counts[i] = fail_counts[i] + 1
    
    return fail_counts / len(test_set)

def getErrorsUpToT(dt, test_set: DataFrame, label_column: str, T:int):
    fail_counts = np.zeros(T)
    for j, row in test_set.iterrows():
        row_dict = row.to_dict()
        predictions:List = dt.getLabelsUpToT(row_dict,T)

        for i in range(0,T):
            if predictions[i] != row_dict[label_column]:
                fail_counts[i] = fail_counts[i] + 1
    
    return fail_counts / len(test_set)

if __name__ == '__main__':
    credit_schema = { \
    "LIMIT_BAL": [] \
        ,"SEX": ['m','f'] \
        ,"EDUCATION": [ ] \
        ,"MARRIAGE": ['y','n','d','u'] \
        ,"AGE": [] \
        ,"PAY_0": []\
        ,"PAY_2": []\
        ,"PAY_3": []\
        ,"PAY_4": []\
        ,"PAY_5": []\
        ,"PAY_6": []\
        ,"BILL_AMT1": []\
        ,"BILL_AMT2": []\
        ,"BILL_AMT3": []\
        ,"BILL_AMT4": []\
        ,"BILL_AMT5": []\
        ,"BILL_AMT6": []\
        ,"PAY_AMT1": []\
        ,"PAY_AMT2": []\
        ,"PAY_AMT3": []\
        ,"PAY_AMT4": []\
        ,"PAY_AMT5": []\
        ,"PAY_AMT6":[] \
        ,"default payment next month": []\
    }

    credit_train = df.read_csv("credit/ccTrain.csv")
    credit_test = df.read_csv("credit/ccTest.csv")

    credit_train.convert_dtypes()
    credit_test.convert_dtypes()
    output_column = "default payment next month"
    
    dt = DecisionTree(credit_train, output_column,credit_schema)
    print("single train error: ", str(getError(dt, credit_train, output_column)))
    print("single test error: ", str(getError(dt, credit_test, output_column)))
    
    endSize = 500
    
    print("started making bagged tree at: ", datetime.now().time())
    bag_of_trees = BaggedTree(credit_train, endSize, output_column, credit_schema)

    print("started making random forest at: ", datetime.now().time())
    rf2 = RandomForest(credit_train, endSize, 2, output_column, credit_schema)

    np_train2_errors = getErrorsUpToT(rf2, credit_train, output_column, endSize)
    np_test2_errors = getErrorsUpToT(rf2, credit_test, output_column, endSize)

    np_train_errors = getErrors(bag_of_trees, credit_train, output_column, endSize)
    np_test_errors = getErrors(bag_of_trees, credit_test, output_column, endSize)

    plt.plot(np_train2_errors)
    plt.plot(np_test2_errors)
    plt.xlabel("trees")
    plt.ylabel("error")
    plt.legend(["training error", "test error"])
    plt.title("credit card random forest error (sample size = 2)")
    plt.show()
    plt.figure()

    plt.plot(range(1,endSize+1), np_train_errors, label="train error")
    plt.plot(range(1,endSize+1), np_test_errors, label="test")
    plt.title("credit card bagged tree errors")
    plt.xlabel("trees")
    plt.ylabel("error")
    plt.legend(["train error", "test error"])
    plt.show()