import numpy
from ..DecisionTree import decisionTree
import pandas as df
import math as m
import numpy as np
from pandas.core.frame import DataFrame

class boostedTree:
    def __getError(self, example_i):
        pass

    def add_iteration(self):

        pass
    
    def __init__(self, training_data: DataFrame, T, output_column: str):
        self.T = T
        self.output_column = output_column
        if T > 500:
            self.alphas = np.zeros(T)
        else:
            self.alphas = np.zeros(500)
        
        #add weights if not already added and set them to 1/Zt
        self.Zt = len(training_data)
        if not "weights" in training_data.columns:
            training_data.insert(0, "weights",  1/self.Zt)
        else:
            training_data["weights"] =  1/self.Zt
        

        self.training_data = training_data
        self.trees = []

        for i in range(0,T):
            add_iteration()




    def getLabel(self, test_data: dict):
        pass