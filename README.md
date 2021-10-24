This is a machine learning library developed by Ted Goodell forCS5350/6350 in University of Utah.

<h1> decision tree </h1>

The decision tree is implemented in the folder decision tree. It contains a file called DT_tester.py
DT_tester.py will evaluate the performance of the decision tree that I wrote which is contained in
decisionTree.py. Specifically it will evaluate the perfomance on six data sets: car test and training
bank test and training, and bank test and training (with 'unknown' values replaced) in that order.

THIS CODE IS TIME CONSUMING! Given the size of this dataset, the DT_tester script can take at least
20 minutes to run.

This decision tree algorithm comes with no warranty implicit or otherwise and should not be relied upon to make
important deicions regarding your career, dating life, or education.

Here is the docstring and method for the decision tree constructor:  


def __init__(self, training_data: DataFrame, label_column:str, schema: dict = None, max_depth: Number = 90, purity_func: Callable[[DataFrame, str, bool], float] = getEntropy):
        Initialize A decision tree object. 

Args:
    training_data (DataFrame): The data that will be used to make this tree. Must have header that matches keys in schema. Numeric columns must have numeric tyoe data.
    label_column (str): The column in training_data that has the output labels
    schema (dict, optional): a dictionary where each key is a column in training_data and corresponds to a list of possible values for that column. Defaults to None.
    max_depth (Number, optional): the maximum depth of this tree. Defaults to 90.
    purity_func (Callable[[DataFrame, str], float], optional): The function that will be used to calculate purity for the information gain 
    (Use getEntropy, getMajorityError, or getGiniIndex contained in decisionTree.py). Defaults to getEntropy.
        
once the tree is constructed, you can get a prediction from it by calling getLabel

def getLabel(self, test_data: dict):
Get the predicted label from the decision tree for test_data
    recursively calls get_label on this tree's root node

Args:
    test_data (dict): a dictionary corresponding to a single row
    in a dataset. The keys must be the same as the column names
    in the training_data DataFrame for this tree. To get such a 
    dictionary from a DataFrame, you can use DataFrame.to_dict()

Returns:
    The predicted label
        
