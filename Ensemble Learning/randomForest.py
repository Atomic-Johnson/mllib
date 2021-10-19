#This is a decision tree library for CS 6350
#Written by Ted Goodell

from typing import Callable
from numpy.core import numeric
from numbers import Number
from numpy.lib.function_base import median
import pandas as df
import math as m
from pandas.core.frame import DataFrame
from pandas.api.types import is_numeric_dtype

DEBUG = False

def getGiniIndex(data_frame: DataFrame, output_column: str, is_weighted: bool = False):
    total = data_frame.shape[0]
    sum = 0
    uniqueVals = data_frame[output_column].unique()
    for value in uniqueVals:
        sum = sum + (len(data_frame.loc[data_frame[output_column] == value])  / total) **2
        # add the proportion squared of each value of the output

    return 1 - sum

def getMajorityError(data_frame: DataFrame, output_column: str, is_weighted: bool = False):

    output_counts = []
    uniqueVals = data_frame[output_column].unique()
    for value in uniqueVals:
        output_counts.append(len(data_frame.loc[data_frame[output_column] == value])) 
        #Gets the number of outputs equal to value

    sum = 0
    majority = 0
    for size in output_counts:
        sum = sum + size

        if size > majority:
            majority = size

    return (sum - majority) / sum


def getEntropy(data_frame: DataFrame, output_column: str, is_weighted: bool):
    if not is_weighted:
        total = len(data_frame)

        entropy = 0
        uniqueVals = data_frame[output_column].unique()
        for value in uniqueVals:
            temp_prop = len(data_frame.loc[data_frame[output_column] == value]) / total
            
            if temp_prop != 0:
                entropy = entropy - temp_prop * m.log2(temp_prop)
    else:
        total = data_frame["weights"].sum()

        entropy = 0
        uniqueVals = data_frame[output_column].unique()
        for value in uniqueVals:
            temp_prop = data_frame.loc[data_frame[output_column] == value]["weights"].sum() / total

            if temp_prop != 0:
                entropy = entropy - temp_prop * m.log2(temp_prop)

    return entropy

def getInfoGain(data_frame: DataFrame, col: str, output_col: str, is_weighted: bool, purity_func: Callable[[DataFrame, str, bool], float]):
    # step 1: get current entropy
    # curr_gain_func = getEntropy(data_frame, output_col)
    curr_purity_factor = purity_func(data_frame, output_col, is_weighted)

    if not is_weighted:
        total = len(data_frame)
    else:
        total = data_frame["weights"].sum()

    # step 2: calculate expected entropy
    expected_purity_factor = 0
    # check if this is numeric
    if is_numeric_dtype(data_frame.dtypes[col]):
        #print(col + " is numeric")
        col_median = data_frame[col].loc[data_frame[col] != -1].median()
        #split on median
        upper_set = data_frame.loc[(data_frame[col] > col_median) & (data_frame[col] != -1)]
        lower_set = data_frame.loc[(data_frame[col] <= col_median) & (data_frame[col] != -1)]
        neg_set = data_frame.loc[data_frame[col] == -1]

        if is_weighted:
            len_neg_set = neg_set["weights"].sum()
            len_upper_set = upper_set["weights"].sum()
            len_lower_set = lower_set["weights"].sum()
        else:
            len_neg_set = len(neg_set)
            len_upper_set = len(upper_set)
            len_lower_set = len(lower_set)

        if len_neg_set != 0:
            expected_purity_factor = expected_purity_factor + neg_set.shape[0]/total * \
                purity_func(neg_set, output_col, is_weighted)

        if len_upper_set != 0:
            expected_purity_factor = expected_purity_factor + upper_set.shape[0]/total * \
                purity_func(upper_set, output_col, is_weighted)

        if len_lower_set != 0:
            expected_purity_factor = expected_purity_factor + lower_set.shape[0]/total * \
                purity_func(lower_set, output_col, is_weighted)

    else: #not numeric do combination analysis
        #print(col + " is not numeric")
        uniqueVals = data_frame[col].unique()
        for value in uniqueVals:
            purity_factor = purity_func(data_frame.loc[data_frame[col] == value], output_col, is_weighted)

            if is_weighted:
                num_Sv = data_frame.loc[data_frame[col] == value]["weights"].sum()
            else:
                num_Sv = len(data_frame.loc[data_frame[col] == value])

            expected_purity_factor = expected_purity_factor + (num_Sv / total) * purity_factor
    
    return curr_purity_factor - expected_purity_factor



class DTPurityFucntions:
    MARJORITY_ERROR = getMajorityError
    ENTROPY = getEntropy
    GINI_INDEX = getGiniIndex

class RandomDecisionTree:
    """Class method to represent a Decision Tree.

        Methods
        -------
        getLabel(test_data: dict):
            returns the label predicted by the decision tree.

    """

    def __init__(self, training_data: DataFrame, label_column:str, schema: dict = None, max_depth: Number = 90, purity_func: Callable[[DataFrame, str, bool], float] = getEntropy):
        """Initialize A decision tree object. 

        Args:
            training_data (DataFrame): The data that will be used to make this tree. Must have header that matches keys in schema. Numeric columns must have numeric tyoe data.
            label_column (str): The column in training_data that has the output labels
            schema (dict, optional): a dictionary where each key is a column in training_data and corresponds to a list of possible values for that column. Defaults to None.
            max_depth (Number, optional): the maximum depth of this tree. Defaults to 90.
            purity_func (Callable[[DataFrame, str], float], optional): The function that will be used to calculate purity for the information gain 
            (Use getEntropy, getMajorityError, or getGiniIndex contained in decisionTree.py). Defaults to getEntropy.
        """
        self.schema: DataFrame = schema
        self.label_column: str = label_column
        self.max_depth: numeric = max_depth
        self.purity_func: Callable[[DataFrame, str, bool], float] = purity_func

        if "weights" in training_data.columns:
            self.is_weighted = True
        else:
            self.is_weighted = False

        self.root = node(training_data, max_depth, self)

    def getLabel(self, test_data: dict):
        """Get the predicted label from the decision tree for test_data
            recursively calls get_label on this tree's root node

        Args:
            test_data (dict): a dictionary corresponding to a single row
            in a dataset. The keys must be the same as the column names
            in the training_data DataFrame for this tree. To get such a 
            dictionary from a DataFrame, you can use DataFrame.to_dict()

        Returns:
            The predicted label
        """
        return self.root.getLabel(test_data)


class node:
        def __init__(self, training_data: DataFrame, max_depth: Number, dt: RandomDecisionTree):
            """Method to create a node from training data. This has been altered to use random forest learning.

            Args:
                training_data (DataFrame): 
                max_depth (Number): 
                dt (DecisionTree): The decision tree that contains the root node and other important constants.
            """
            

            # step 1: should this be a leaf node?
            unique_labels = training_data[dt.label_column].unique()
            if len(unique_labels) == 1 or max_depth == 0 or training_data.shape[1] == 1:
                self.is_leaf = True
                #majority_amount = 0
                self.label = training_data[dt.label_column].mode()[0]
                # for label in unique_labels:
                #     amount = len(training_data.loc[training_data[dt.label_column] == label])
                #     training_data[dt.label_column].m
                #     if amount > majority_amount:
                #         majority_amount = amount
                #         self.label = label
                 
                if DEBUG:
                    if self.label is None:
                        print("WARNING: No label assigned to leaf node")
                    print(" Leaf Node with label: " + str(self.label))

                return
            else:
                self.is_leaf = False

            # step 2: Determine which node to use to split
            randomSelection = training_data.sample(frac=0.1, replace=False, axis='columns')
            if randomSelection.isnull():
                randomSelection = training_data.sample(frac=0.1, replace=False, axis='columns')
                
            max_gain = -1
            
            for  attribute in randomSelection.columns:
                if attribute == dt.label_column or attribute == "weights":
                    continue

                gain = getInfoGain(randomSelection, attribute, dt.label_column, dt.is_weighted, dt.purity_func)
                if gain > max_gain:
                    max_gain = gain
                    self.best_attribute = attribute
            
            if DEBUG:
                print(" Node on: " + str(self.best_attribute), end="")

            # step 3: make children
            self.children = {}
            if is_numeric_dtype(training_data.dtypes[self.best_attribute]):
                self.isNumeric = True
                self.medianSplit = training_data[self.best_attribute].loc[training_data[self.best_attribute] != -1].median()

                child_frame = training_data.loc[training_data[self.best_attribute] > self.medianSplit].drop(self.best_attribute, axis=1)
                if len(child_frame) != 0:
                    self.children["gt"] = node(child_frame, max_depth -1, dt)
                else: #leaf node
                    self.children["gt"] = node(training_data, 0, dt)

                child_frame = training_data.loc[(training_data[self.best_attribute] <= self.medianSplit) & (training_data[self.best_attribute] != -1)].drop(self.best_attribute, axis=1)
                if len(child_frame) != 0:
                    self.children["lt"] = node(child_frame, max_depth -1, dt)
                else: #leaf node
                    self.children["lt"] = node(training_data, 0, dt)

                child_frame = training_data.loc[training_data[self.best_attribute] == -1].drop(self.best_attribute, axis=1)
                if len(child_frame) != 0:
                    self.children["neg"] = node(child_frame, max_depth -1, dt)
                else: #leaf node
                    self.children["neg"] = node(training_data, 0, dt)

            else:
                uniqueTrainVals = training_data[self.best_attribute].unique()
                self.isNumeric = False
                for value in dt.schema[self.best_attribute]:
                    if DEBUG:
                        print(" val:" + str(value), end="")

                    if not value in uniqueTrainVals:
                        # this value was not present in this subset so we need to make a leaf node
                        self.children[value] = node(training_data, 0, dt)
                    else:
                        child_frame = training_data.loc[training_data[self.best_attribute] == value].drop(self.best_attribute, axis=1)
                        self.children[value] = node(child_frame, max_depth -1, dt)
                

        def getLabel(self, test_data: dict):
            if self.is_leaf:
                return self.label

            if self.isNumeric:
                if test_data[self.best_attribute] == -1:
                    return self.children["neg"].getLabel(test_data)

                if test_data[self.best_attribute] > self.medianSplit:
                    return self.children["gt"].getLabel(test_data)
                else:
                    return self.children["lt"].getLabel(test_data)
                print("WARNING: numeric data did not go to numeric node")
            # get the value for this attribute
            value = test_data[self.best_attribute]
            return self.children[value].getLabel(test_data)

class RandomForest:
    def __init__(self, training_data: DataFrame, T, output_column: str, schema: dict) -> None:
        self.T = 0