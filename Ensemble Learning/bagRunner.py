import sys
import os
import pandas as df
from pandas.core.frame import DataFrame
from treeBagger import BaggedTree

# for arguments this expects bagRunner.py <startSize> <endSize> <outputFile.csv>
# example: $ python bagRunner.py 1 350 bag1.csv
def getError(dt, test_set: DataFrame, label_column: str):
    fail_count = 0.0
    for i, row in test_set.iterrows():
        row_dict = row.to_dict()
        prediction = dt.getLabel(row_dict)

        if prediction != row_dict[label_column]:
            fail_count = fail_count + 1
    
    return fail_count / len(test_set)


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
stump_results = df.DataFrame(columns=["stump", "train error", "test error"],dtype=float)

# The adaboost algorithm assumes output of -1 or 1
bank_frame_train["y"] = bank_frame_train["y"].replace("no", -1)
bank_frame_train["y"] = bank_frame_train["y"].replace("yes", 1)
bank_frame_test["y"] = bank_frame_train["y"].replace("no", -1)
bank_frame_test["y"] = bank_frame_train["y"].replace("yes", 1)

bag_of_trees = BaggedTree(bank_frame_train, startSize - 1, 'y', bank_schema)

for i in range(startSize, endSize + 1):
    print(i, end=" ", flush=True)
    bag_of_trees.add_iteration()
    iteration_results.append({"iteration":i, \
         "train error":getError(bag_of_trees,bank_frame_train,"y"), "test error":getError(bag_of_trees,bank_frame_test,"y") }, ignore_index=True)

iteration_results.to_csv(outFile)

print("finished run " + str(startSize) + " " + str(endSize) + " " + outFile)