from typing import List
import numpy
from ..DecisionTree import decisionTree as dt
import pandas as df
import math as m
import numpy as np
from pandas.core.frame import DataFrame

MAX_TREE_DEPTH = 2
WEIGHTS = 'weights'
DEBUG = True

class boostedTree:
    def __getError(self, tree: dt.DecisionTree):
        total = len(self.training_data)
        sum = 0.0

        for i, row in self.training_data.iterrows():
            sum = sum + row[WEIGHTS] * row[self.output_column] * tree.getLabel(row.to_dict())

        return 0.5 - 0.5 * sum

    def add_iteration(self):
        #step one build a new tree
        temp_tree = dt.DecisionTree(self.training_data, self.output_column, self.schema, MAX_TREE_DEPTH)

        #step 2 get error and alpha
        Et = self.__getError(temp_tree)
        alpha = 0.5 *  m.log((1-Et) / Et)
        
        #step 3 update weights
        for i, row in self.training_data.iterrows():
            Dt = row[WEIGHTS]
            self.training_data.iloc(i)[WEIGHTS] = Dt/self.Zt * \
                m.exp(-1 * alpha * temp_tree.getLabel(row.to_dict) * row[self.output_column])

        #step 4 add new tree and alpha to list
        self.alphas.append(alpha)
        self.trees.append(temp_tree)
        self.T = self.T + 1

        if DEBUG:
            if self.training_data[WEIGHTS].sum() != 1:
                print("ERROR: weights did not add up to one")
    
    def __init__(self, training_data: DataFrame, T, output_column: str, schema: dict):
        self.T = 0
        self.output_column = output_column
        self.schema = schema
        self.alphas = []

        # if T > 500:
        #     self.alphas = np.zeros(T)
        # else:
        #     self.alphas = np.zeros(500)
        
        #add weights if not already added and set them to 1/Zt
        self.Zt = len(training_data)
        if not WEIGHTS in training_data.columns:
            training_data.insert(0, WEIGHTS,  1/self.Zt)
        else:
            training_data[WEIGHTS] =  1/self.Zt
        

        self.training_data = training_data
        self.trees: List(dt.DecisionTree) = []

        for i in range(0,T):
            self.add_iteration()


    def getLabel(self, test_data: dict):
        average_label = 0.0
        for i, tree in self.trees:
            average_label = average_label + self.alphas[i] * tree.getLabel(test_data)

        if average_label >= 0:
            return 1
        else:
            return -1