import numpy as np
import pandas as df
from pandas.core.frame import DataFrame
from pandas.core.series import Series

class Averaged_Perceptron():

    def add_epoch(self):
        shuffle_data = self.training_data.sample(len(self.training_data), replace=False)

        for i, row in shuffle_data.iterrows():
            output = row[self.output_column]
            x = row.drop(self.output_column).to_numpy()
            
            if self.w.T.dot(x) * output <= 0:
                self.w = self.w + self.r * output * x
            
            self.a = self.a + self.w

    def __init__(self, training_data: DataFrame, output_column: str, epochs, r = 0.5) -> None:
        self.training_data = training_data.copy(deep=True)
        #Let's wrap in a bias term
        self.training_data.insert(0, "bias", 1)
        
        self.output_column = output_column
        self.r = r
        self.w: np.ndarray = np.zeros(len(self.training_data.columns) - 1)
        self.a: np.ndarray = self.w.copy() 

        for i in range(epochs):
            self.add_epoch()

    def get_label(self, row: Series):
        # row needs to be augmented to support the bias.
        row["bias"] = 1
        output = self.a.T.dot(row.drop(self.output_column).to_numpy())

        if output >= 0:
            return 1
        else:
            return -1