from typing import List
from decisionTree import DecisionTree
import pandas as df
import math as m
import numpy as np
from pandas.core.frame import DataFrame

class BaggedTree:

    def add_iteration(self):
        bootstrap = self.training_data.sample(self.sample_size, replace=True)
        self.trees.append(DecisionTree(bootstrap, self.output_column, self.schema))
        self.T = self.T + 1
        
    def __init__(self, training_data: DataFrame, T, output_column: str, schema: dict):
        self.T = 0
        self.sample_size = len(training_data)
        self.output_column = output_column
        self.schema = schema
        self.training_data: DataFrame = training_data
        self.trees:List[DecisionTree] = []

        for i in range(0, T):
            self.add_iteration()

    def getLabel(self, test_data: dict):
        sum = 0

        for tree in self.trees:
            sum = sum + tree.getLabel(test_data)

        if sum >= 0:
            return 1
        else:
            return -1
        