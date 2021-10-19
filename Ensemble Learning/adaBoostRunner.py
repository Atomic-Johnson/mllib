import pandas as df
from adaboost import boostedTree
from pandas.core.frame import DataFrame

def getError(dt, test_set: DataFrame, label_column: str):
    fail_count = 0.0
    for i, row in test_set.iterrows():
        row_dict = row.to_dict()
        prediction = dt.getLabel(row_dict)

        if prediction != row_dict[label_column]:
            fail_count = fail_count + 1
    
    return fail_count / len(test_set)

bank_frame_train = df.read_csv("bank/train.csv", names=["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", \
        "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"])

bank_frame_test = df.read_csv("bank/test.csv", names=["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", \
    "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"])


bank_frame_train["y"] = bank_frame_train["y"].replace("no", -1)
bank_frame_train["y"] = bank_frame_train["y"].replace("yes", 1)
bank_frame_test["y"] = bank_frame_train["y"].replace("no", -1)
bank_frame_test["y"] = bank_frame_train["y"].replace("yes", 1)
bank_frame_train.apply(df.to_numeric, errors="ignore")
bank_frame_test.apply(df.to_numeric, errors="ignore")

bank_frame_test.to_excel("yeah.xlsx")

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


booster = boostedTree(bank_frame_train, 0, "y", bank_schema)
print("starting run")
for i in range(1, 501):
    print(i, end=" ", flush=True)
    booster.add_iteration()
    iteration_results.append({"iteration":i, \
         "train error":getError(booster,bank_frame_train,"y"), "test error":getError(booster,bank_frame_test,"y") }, ignore_index=True)

print("Evaluating stumps")
for i, tree in zip(range(1,501), booster.trees):
    print(i, end=" ", flush=True)
    stump_results.append({"stump":i, "train error":getError(tree,bank_frame_train,"y"), \
        "test error":getError(tree,bank_frame_test,"y") }, ignore_index=True)

iteration_results.to_excel("iteration_errors.xlsx")
stump_results.to_excel("stump_errors.xlsx")


