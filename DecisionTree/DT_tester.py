
from pandas.core.frame import DataFrame
from decisionTree import DecisionTree, getMajorityError, getGiniIndex
import pandas as df

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


tennis_frame = df.read_csv("tennis.csv")
car_frame_train = df.read_csv("train.csv", names=["buying", "maint", "doors", "persons", "lug_boot", "safety", "label"])
car_frame_test = df.read_csv("test.csv", names=["buying", "maint", "doors", "persons", "lug_boot", "safety", "label"])

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
tennis_tree = DecisionTree(tennis_frame, label_column="play", purity_func=getMajorityError, schema=tennis_schema)

'''
for i in range(1,7):
    car_tree = DecisionTree(car_frame_train, label_column="label", purity_func=getGiniIndex, schema=car_schema, max_depth=i)
    ginAccTrain = getAccuracy(car_tree, car_frame_train, 'label')
    ginAccTest = getAccuracy(car_tree, car_frame_test, 'label')

    car_tree = DecisionTree(car_frame_train, label_column="label", schema=car_schema, max_depth=i) #purity function defaults to getEntropy
    entAccTrain = getAccuracy(car_tree, car_frame_train, 'label')
    entAccTest = getAccuracy(car_tree, car_frame_test, 'label')

    car_tree = DecisionTree(car_frame_train, label_column="label", purity_func=getMajorityError, schema=car_schema, max_depth=i)
    majErrAccTrain = getAccuracy(car_tree, car_frame_train, 'label')
    majErrAccTest = getAccuracy(car_tree, car_frame_test, 'label')

    print(str(i) + " ginTrain: " + str(ginAccTrain) + " ginTest: " + str(ginAccTest) + \
         " entTrain: " + str(entAccTrain) + " entTest: " + str(entAccTest) + \
         " majErrTrain: " + str(majErrAccTrain) + " majErrTest: " + str(majErrAccTest))
'''

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
    bank_frame_train[col].apply(doReplacement)

for col in bank_frame_test:
    most_common = ''
    most_qty = 0
    for value in bank_frame_test[col].unique():
        qty = bank_frame_test.loc[bank_frame_test[col] == value].shape[0]
        if qty > most_qty:
            most_qty = qty
            most_common = value
    
    replacement = most_common
    bank_frame_test[col].apply(doReplacement)

print("finished replacements")

for i in range(1,17):

    bank_tree = DecisionTree(bank_frame_train, label_column="y", schema=bank_schema, max_depth=i) #purity function defaults to getEntropy
    entAccTrain = getError(bank_tree, bank_frame_train, 'y')
    entAccTest = getError(bank_tree, bank_frame_test, 'y')
    
    bank_tree = DecisionTree(bank_frame_train, label_column="y", purity_func=getGiniIndex, schema=bank_schema, max_depth=i)
    ginAccTrain = getError(bank_tree, bank_frame_train, 'y')
    ginAccTest = getError(bank_tree, bank_frame_test, 'y')

    bank_tree = DecisionTree(bank_frame_train, label_column="y", purity_func=getMajorityError, schema=bank_schema, max_depth=i)
    majErrAccTrain = getError(bank_tree, bank_frame_train, 'y')
    majErrAccTest = getError(bank_tree, bank_frame_test, 'y')

    print(str(i) + " ginTrain: " + str(ginAccTrain) +\
         " entTrain: " + str(entAccTrain) +
         " majErrTrain: " + str(majErrAccTrain))

    print(str(i) + " ginTest: " + str(ginAccTest) + \
         " entTest: " + str(entAccTest) + \
          " majErrTest: " + str(majErrAccTest))

exit()

for i,row in tennis_frame.iterrows():
    print(row.to_dict())
    print("prediction: " + str(car_tree.getLabel(row.to_dict())))

print("wait here")