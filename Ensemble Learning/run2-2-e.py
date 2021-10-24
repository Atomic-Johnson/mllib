import sys
import os
from typing import List
import pandas as df
import multiprocessing as mp
import numpy as np
from pandas.core.frame import DataFrame
from randomForest import RandomForest
from datetime import datetime

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

    print("started making trees at: ", datetime.now().time())
    tree_bags = []
    single_trees = []
    for i in range(100):
        bag_sample = bank_frame_train.sample(1000, replace=False)
        treebag = RandomForest(bag_sample, 500, 2, 'y', bank_schema)
        tree_bags.append(treebag)
        single_trees.append(treebag.trees[0])

    print("finished making trees at: ", datetime.now().time())
    predictions_array = np.zeros(100)
    bias_terms = np.zeros(len(bank_frame_test))
    variances = np.zeros(len(bank_frame_test))
    for i, test in bank_frame_test.iterrows():
        test_dict = test.to_dict()

        for j, tree in zip(range(100), single_trees):
            predictions_array[j] = tree.getLabel(test_dict)

        mean_prediction:float = predictions_array.mean()
        bias_terms[i] = (mean_prediction - test_dict['y'])**2

        variances[i] = 1/99 * ((predictions_array - mean_prediction)**2).sum()


    print("Average bias for single tree learner = " + str(bias_terms.mean()) + " Average variance for single tree learner = " + str(variances.mean()))

    for i, test in bank_frame_test.iterrows():
        test_dict = test.to_dict()

        for j, tree in zip(range(100), tree_bags):
            predictions_array[j] = tree.getLabel(test_dict)

        mean_prediction:float = predictions_array.mean()
        bias_terms[i] = (mean_prediction - test_dict['y'])**2

        variances[i] = 1/99 * ((predictions_array - mean_prediction)**2).sum()

    print("Average bias for bagged learner = " + str(bias_terms.mean()) + " Average variance for bagged learner = " + str(variances.mean()))

    print("finished script at: ", datetime.now().time())