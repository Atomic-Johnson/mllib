import sys
import os
from typing import List
import pandas as df
import multiprocessing as mp
import numpy as np
from pandas.core.frame import DataFrame
from randomForest import RandomForest
from datetime import datetime
import matplotlib.pyplot as plt

# for arguments this expects bagRunner.py <startSize> <endSize> <outputFile.csv>
# example: $ python bagRunner.py 1 350 bag1.csv
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

    iteration_results2 = df.DataFrame(columns=["iteration", "train error", "test error"],dtype=float)
    iteration_results4 = df.DataFrame(columns=["iteration", "train error", "test error"],dtype=float)
    iteration_results6 = df.DataFrame(columns=["iteration", "train error", "test error"],dtype=float)
    print("started making trees at: ", datetime.now().time())

    rf2 = RandomForest(bank_frame_train, 500, 2, 'y', bank_schema)
    rf4 = RandomForest(bank_frame_train, 500, 4, 'y', bank_schema)
    rf6 = RandomForest(bank_frame_train, 500, 6, 'y', bank_schema)
    print("finished making trees at: ", datetime.now().time())

    np_train2_errors = getErrorsUpToT(rf2, bank_frame_train, 'y', 500)
    np_test2_errors = getErrorsUpToT(rf2, bank_frame_test, 'y', 500)

    np_train4_errors = getErrorsUpToT(rf4, bank_frame_train, 'y', 500)
    np_test4_errors = getErrorsUpToT(rf4, bank_frame_test, 'y', 500)

    np_train6_errors = getErrorsUpToT(rf6, bank_frame_train, 'y', 500)
    np_test6_errors = getErrorsUpToT(rf6, bank_frame_test, 'y', 500)


    for i in range(500):
        iteration_results2 = iteration_results2.append({"iteration": i+1, \
                "train error":np_train2_errors[i], "test error":np_test2_errors[i] }, ignore_index=True)

        iteration_results4 = iteration_results4.append({"iteration": i+1, \
                "train error":np_train4_errors[i], "test error":np_test4_errors[i] }, ignore_index=True)

        iteration_results6 = iteration_results6.append({"iteration": i+1, \
                "train error":np_train6_errors[i], "test error":np_test6_errors[i] }, ignore_index=True)
        
    iteration_results2.to_csv("rf-2.csv")
    iteration_results4.to_csv("rf-4.csv")
    iteration_results6.to_csv("rf-6.csv")

    print("Errors calculated and saved at: ", datetime.now().time())

    plt.plot(np_train2_errors)
    plt.plot(np_test2_errors)
    plt.legend(["training error", "test error"])
    plt.title("random forest error (sample size = 2)")
    plt.show()
    plt.figure()

    plt.plot(np_train4_errors)
    plt.plot(np_test4_errors)
    plt.legend(["training error", "test error"])
    plt.title("random forest error (sample size = 4)")
    plt.show()
    plt.figure()

    plt.plot(np_train6_errors)
    plt.plot(np_test6_errors)
    plt.legend(["training error", "test error"])
    plt.title("random forest error (sample size = 6)")
    plt.show()
    

    

