# Ted Gooodell
# This is a script for getting test results for my Decision Tree algorithm

from typing import List
from pandas.core.frame import DataFrame
from decisionTree import DecisionTree, getMajorityError, getGiniIndex, getEntropy
import pandas as df
#import threading as th
import multiprocessing as pr

def getError(dt: DecisionTree, test_set: DataFrame, label_column: str):
    pass_count = 0.0
    fail_count = 0.0
    for i, row in test_set.iterrows():
        row_dict = row.to_dict()
        prediction = dt.getLabel(row_dict)

        if prediction == row_dict[label_column]:
            pass_count = pass_count + 1
        else:
            fail_count = fail_count + 1
    
    return fail_count / (pass_count + fail_count)

def testPerformance(train_set: DataFrame, test_set: DataFrame, depth, schema:dict, purity_func, purity_str: str, label_col: str, lock ):
    tree = DecisionTree(train_set,label_col, schema, depth, purity_func)
    train_error = getError(tree, train_set, label_col)
    test_error = getError(tree, test_set, label_col)

    lock.acquire()
    print(str(depth) + " " + purity_str + "\t" + str(train_error) + "\t" + str(test_error), flush=True)
    lock.release()

####################################### Start of Main Script ##########################################
if __name__ == '__main__':
    pr.freeze_support()

    #tennis_frame = df.read_csv("tennis.csv")

    # Frames passed into DecisionTree must contain a header that matches a schema
    car_frame_train = df.read_csv("car/train.csv", names=["buying", "maint", "doors", "persons", "lug_boot", "safety", "label"])
    car_frame_test = df.read_csv("car/test.csv", names=["buying", "maint", "doors", "persons", "lug_boot", "safety", "label"])

    bank_frame_train = df.read_csv("bank/train.csv", names=["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", \
        "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"])

    bank_frame_test = df.read_csv("bank/test.csv", names=["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", \
        "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"])

    #Convert these data frames to numeric
    bank_frame_train.apply(df.to_numeric, errors="ignore")
    bank_frame_test.apply(df.to_numeric, errors="ignore")



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

    car_schema = {'buying':['vhigh', 'high', 'med', 'low'], \
        'maint':['vhigh', 'high', 'med', 'low'], \
        'doors':['2', '3', '4','5more'], \
        'persons':['2', '4', 'more'], \
        'lug_boot':['small', 'med', 'big'], \
        'safety':['low', 'med', 'high'], \
        'label':['unacc', 'acc', 'good', 'vgood']}

    tennis_schema = {'O': ['S','O','R'], 'T': ['H', 'M', 'C'], 'H': ['H', 'N', 'L'], 'W': ['S', 'W'], 'play': ['0', '1']}

    car_tree = DecisionTree(car_frame_train, label_column="label", purity_func=getGiniIndex, schema=car_schema)
    #tennis_tree = DecisionTree(tennis_frame, label_column="play", purity_func=getMajorityError, schema=tennis_schema)

    
    for i in range(1,7):
        car_tree = DecisionTree(car_frame_train, label_column="label", purity_func=getGiniIndex, schema=car_schema, max_depth=i)
        ginAccTrain = getError(car_tree, car_frame_train, 'label')
        ginAccTest = getError(car_tree, car_frame_test, 'label')

        car_tree = DecisionTree(car_frame_train, label_column="label", schema=car_schema, max_depth=i) #purity function defaults to getEntropy
        entAccTrain = getError(car_tree, car_frame_train, 'label')
        entAccTest = getError(car_tree, car_frame_test, 'label')

        car_tree = DecisionTree(car_frame_train, label_column="label", purity_func=getMajorityError, schema=car_schema, max_depth=i)
        majErrAccTrain = getError(car_tree, car_frame_train, 'label')
        majErrAccTest = getError(car_tree, car_frame_test, 'label')

        print(str(i) + " ginTrain: " + str(ginAccTrain) + " ginTest: " + str(ginAccTest) + \
            " entTrain: " + str(entAccTrain) + " entTest: " + str(entAccTest) + \
            " majErrTrain: " + str(majErrAccTrain) + " majErrTest: " + str(majErrAccTest))
    

    print("\t running bank set with unknown labels")
    print("\t \t Train Set \t Test Set")

    processes: List[pr.Process] = []
    locker = pr.Lock()
    for i in range(2,17,2):
        processes.append(pr.Process(testPerformance(bank_frame_train, bank_frame_test, i - 1, bank_schema, getEntropy, "Entropy", "y", locker)))
        processes.append(pr.Process(testPerformance(bank_frame_train, bank_frame_test, i - 1, bank_schema, getGiniIndex, "GiniIndex", "y", locker)))
        processes.append(pr.Process(testPerformance(bank_frame_train, bank_frame_test, i - 1, bank_schema, getEntropy, "MajorityErr", "y", locker)))
        
        processes.append(pr.Process(testPerformance(bank_frame_train, bank_frame_test, i, bank_schema, getEntropy, "Entropy", "y", locker)))
        processes.append(pr.Process(testPerformance(bank_frame_train, bank_frame_test, i, bank_schema, getGiniIndex, "GiniIndex", "y", locker)))
        processes.append(pr.Process(testPerformance(bank_frame_train, bank_frame_test, i, bank_schema, getEntropy, "MajorityErr", "y", locker)))

        for proc in processes:
            proc.start()

        for proc in processes:
            proc.join()

        processes.clear()

        
    replacement = ''

    def doReplacement(orig):
        if orig == 'unknown':
            return replacement
        else:
            return orig

    #Do unknown replacement
    for col in bank_frame_train:
        most_common = ''
        most_qty = 0
        for value in bank_frame_train[col].unique():
            qty = bank_frame_train.loc[bank_frame_train[col] == value].shape[0]
            if qty > most_qty:
                most_qty = qty
                most_common = value
        
        replacement = most_common
        bank_frame_train[col] = bank_frame_train[col].apply(doReplacement)

    for col in bank_frame_test:
        most_common = ''
        most_qty = 0
        for value in bank_frame_test[col].unique():
            qty = bank_frame_test.loc[bank_frame_test[col] == value].shape[0]
            if qty > most_qty:
                most_qty = qty
                most_common = value
        
        replacement = most_common
        bank_frame_test[col] = bank_frame_test[col].apply(doReplacement)

    print("finished replacements")

    print("\t running bank set with unknown labels replaced")
    print("\t \t Train Set \t Test Set")

    processes: List[pr.Process] = []
    locker = pr.Lock()
    for i in range(2,17,2):
        processes.append(pr.Process(testPerformance(bank_frame_train, bank_frame_test, i - 1, bank_schema, getEntropy, "Entropy", "y", locker)))
        processes.append(pr.Process(testPerformance(bank_frame_train, bank_frame_test, i - 1, bank_schema, getGiniIndex, "GiniIndex", "y", locker)))
        processes.append(pr.Process(testPerformance(bank_frame_train, bank_frame_test, i - 1, bank_schema, getEntropy, "MajorityErr", "y", locker)))
        
        processes.append(pr.Process(testPerformance(bank_frame_train, bank_frame_test, i, bank_schema, getEntropy, "Entropy", "y", locker)))
        processes.append(pr.Process(testPerformance(bank_frame_train, bank_frame_test, i, bank_schema, getGiniIndex, "GiniIndex", "y", locker)))
        processes.append(pr.Process(testPerformance(bank_frame_train, bank_frame_test, i, bank_schema, getEntropy, "MajorityErr", "y", locker)))

        for proc in processes:
            proc.start()

        for proc in processes:
            proc.join()

        processes.clear()
    exit()