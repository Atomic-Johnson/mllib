import numpy as np
import pandas as df
import math as m
from pandas.core.frame import DataFrame
from pandas.core.series import Series

class Voted_Perceptron():

    def add_epoch(self):
        shuffle_data = self.training_data.sample(len(self.training_data), replace=False)

        for i, row in shuffle_data.iterrows():
            output = row[self.output_column]
            x = row.drop(self.output_column).to_numpy()
            

            if self.w.T.dot(x) * output <= 0:
                self.w = self.w + self.r * output * x
                self.wi.append(self.w)
                self.m = self.m + 1
                self.Cm.append(1)
            else:
                self.Cm[self.m - 1] = self.Cm[self.m - 1] + 1

    def __init__(self, training_data: DataFrame, output_column: str, epochs, r = 0.5) -> None:
        self.training_data = training_data.copy(deep=True)
        #Let's wrap in a bias term
        self.training_data.insert(0, "bias", 1)
        
        self.output_column = output_column
        self.r = r
        self.m = 0
        self.Cm = []
        self.w: np.ndarray = np.zeros(len(self.training_data.columns) - 1) 
        self.wi = []

        for i in range(epochs):
            self.add_epoch()

    def _get_sign(self, number):
        if number >= 0:
            return 1
        else:
            return -1

    def get_label(self, row: Series):
        # row needs to be augmented to support the bias.
        row["bias"] = 1
        x = row.drop(self.output_column).to_numpy()
        sum = 0.0

        if len(self.Cm) != len(self.wi):
            print("Cm is not the same length as Wi! I did it wrong.")
        
        for c, w in zip(self.Cm, self.wi):
            sum = sum + c * self._get_sign(w.T.dot(x))

        return self._get_sign(sum)