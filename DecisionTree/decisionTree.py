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

def getGiniIndex(data_frame: DataFrame, output_column: str):
    total = data_frame.shape[0]
    sum = 0
    for value in data_frame[output_column].unique():
        sum = sum + (data_frame.loc[data_frame[output_column] == value].shape[0]  / total) **2
        # add the proportion squared of each value of the output

    return 1 - sum

def getMajorityError(data_frame: DataFrame, output_column: str):

    output_counts = []
    for value in data_frame[output_column].unique():
        output_counts.append(data_frame.loc[data_frame[output_column] == value].shape[0]) 
        #Gets the number of outputs equal to value

    sum = 0
    majority = 0
    for size in output_counts:
        sum = sum + size

        if size > majority:
            majority = size

    return (sum - majority) / sum


def getEntropy(data_frame: DataFrame, output_column: str):
    total = data_frame.shape[0]

    entropy = 0
    for value in data_frame[output_column].unique():
        temp_prop = data_frame.loc[data_frame[output_column] == value].shape[0] / total
        
        if temp_prop != 0:
            entropy = entropy - temp_prop * m.log2(temp_prop)

    return entropy

def getInfoGain(data_frame: DataFrame, col: str, output_col: str, purity_func: Callable[[DataFrame, str], float]):
    # step 1: get current entropy
    # curr_gain_func = getEntropy(data_frame, output_col)
    curr_purity_factor = purity_func(data_frame, output_col)
    total = data_frame.shape[0]

    # step 2: calculate expected entropy
    expected_purity_factor = 0
    # check if this is numeric
    if is_numeric_dtype(data_frame[col]):
        #print(col + " is numeric")
        col_median = data_frame[col].loc[data_frame[col] != -1].median()
        #split on median
        upper_set = data_frame.loc[(data_frame[col] > col_median) & (data_frame[col] != -1)]
        lower_set = data_frame.loc[(data_frame[col] <= col_median) & (data_frame[col] != -1)]
        neg_set = data_frame.loc[data_frame[col] == -1]

        if neg_set.shape[0] != 0:
            expected_purity_factor = expected_purity_factor + neg_set.shape[0]/total * \
                purity_func(neg_set, output_col)

        if upper_set.shape[0] != 0:
            expected_purity_factor = expected_purity_factor + upper_set.shape[0]/total * \
                purity_func(upper_set, output_col)

        if lower_set.shape[0] != 0:
            expected_purity_factor = expected_purity_factor + lower_set.shape[0]/total * \
                purity_func(lower_set, output_col)

    else: #not numeric do combination analysis
        #print(col + " is not numeric")
        for value in data_frame[col].unique():
            purity_factor = purity_func(data_frame.loc[data_frame[col] == value], output_col)

            expected_purity_factor = expected_purity_factor + \
            data_frame.loc[data_frame[col] == value].shape[0]/total * purity_factor
    
    return curr_purity_factor - expected_purity_factor



class DTPurityFucntions:
    MARJORITY_ERROR = getMajorityError
    ENTROPY = getEntropy
    GINI_INDEX = getGiniIndex

class DecisionTree:
    """Class method to create DecisionTree .
    """
    def __init__(self, training_data: DataFrame, label_column:str, schema: dict = None, max_depth: Number = 90, purity_func: Callable[[DataFrame, str], float] = getEntropy):
        self.schema: DataFrame = schema
        self.label_column: str = label_column
        self.max_depth: numeric = max_depth
        self.purity_func: Callable[[DataFrame, str], float] = purity_func

        self.root = node(training_data, max_depth, self)

    def getLabel(self, test_data: dict):
        return self.root.getLabel(test_data)

class node:
        def __init__(self, training_data: DataFrame, max_depth: Number, dt: DecisionTree):
            # step 1: should this be a leaf node?
            self.is_leaf = False
            if training_data[dt.label_column].unique().shape[0] == 1 or max_depth == 0 or training_data.shape[1] == 1:
                self.is_leaf = True
                majority_amount = 0
                
                for label in training_data[dt.label_column].unique():
                    amount = training_data.loc[training_data[dt.label_column] == label].shape[0]
                    if amount > majority_amount:
                        majority_amount = amount
                        self.label = label
                 
                if DEBUG:
                    if self.label is None:
                        print("WARNING: No label assigned to leaf node")
                    print(" Leaf Node with label: " + str(self.label))

                return

            # step 2: Determine which node to use to split
            max_gain = -1
            
            for attribute in training_data.columns:
                if attribute == dt.label_column:
                    continue

                gain = getInfoGain(training_data, attribute, dt.label_column, dt.purity_func)
                if gain > max_gain:
                    max_gain = gain
                    self.best_attribute = attribute
            
            if DEBUG:
                print(" Node on: " + str(self.best_attribute), end="")

            # step 3: make children
            self.children = {}
            if is_numeric_dtype(training_data[self.best_attribute]):
                self.isNumeric = True
                self.medianSplit = training_data[self.best_attribute].loc[training_data[self.best_attribute] != -1].median()

                child_frame = training_data.loc[training_data[self.best_attribute] > self.medianSplit].drop(self.best_attribute, axis=1)
                if child_frame.shape[0] != 0:
                    self.children["gt"] = node(child_frame, max_depth -1, dt)

                child_frame = training_data.loc[(training_data[self.best_attribute] <= self.medianSplit) & (training_data[self.best_attribute] != -1)].drop(self.best_attribute, axis=1)
                if child_frame.shape[0] != 0:
                    self.children["lt"] = node(child_frame, max_depth -1, dt)

                child_frame = training_data.loc[training_data[self.best_attribute] == -1]
                if child_frame.shape[0] != 0:
                    self.children["neg"] = node(child_frame, max_depth -1, dt)

            else:
                self.isNumeric = False
                for value in dt.schema[self.best_attribute]:
                    if DEBUG:
                        print(" val:" + str(value), end="")

                    if not value in training_data[self.best_attribute].unique():
                        # this value was not present in this subset so we need to make a leaf node
                        self.children[value] = node(training_data, 0, dt)
                    else:
                        child_frame = training_data.loc[training_data[self.best_attribute] == value].drop(self.best_attribute, axis=1)
                        self.children[value] = node(child_frame, max_depth -1, dt)

                # step 4: add leaf nodes for values not in our set of 
                

        def getLabel(self, test_data: dict):
            if self.is_leaf:
                return self.label

            if self.isNumeric:
                if test_data[self.best_attribute] == -1 and "neg" in test_data.keys():
                    return self.children["neg"].getLabel(test_data)

                if test_data[self.best_attribute] > self.medianSplit and "gt" in test_data.keys():
                    return self.children["gt"].getLabel(test_data)
                else:
                    if "lt" in test_data.keys():
                        return self.children["lt"].getLabel(test_data)
                    elif "neg" in test_data.keys():
                        return self.children["neg"].getLabel(test_data)
                return 'yes'
            # get the value for this attribute
            value = test_data[self.best_attribute]
            return self.children[value].getLabel(test_data)




