import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from pandas.core.series import Series

class PrimalSVM():

    def gammaA(self, t):
        gamma0 = 0.5
        a = 0.1

        return gamma0 / (1 + gamma0/a * t)

    def gammaB(self, t):
        gamma0 = 0.5
        return gamma0 / (1 + t)

    def add_epoch(self):
        #xi = self.training_data.sample(1)
        shuffle_data = self.training_data.sample(len(self.training_data), replace=False)
        self.epochs = self.epochs + 1

        #w0 should be the weight vector without the bias.
        #w0 = self.w[0:-1]
        for i,row in shuffle_data.iterrows():
            output = row[self.output_column]
            xi = row.drop(self.output_column).to_numpy()

            # the last term is the bias so we will represent w0 with w[0:-1]
            if self.w.T.dot(xi) * output <= 1:
                gammat = self.gammaFunc(self.epochs)
                self.w = self.w - gammat * np.append(self.w[0:-1], 0) + (self.C * output * len(shuffle_data) * gammat) * xi
            else:
                self.w[0:-1] = (1 - self.gammaFunc(self.epochs)) * self.w[0:-1]
                 

        

    def __init__(self, training_data: DataFrame, output_column: str, epochs, C = 0.5, gammaFunc = 'A', do_objective_output = False) -> None:
        self.training_data = training_data.copy(deep=True)
        #Let's wrap in a bias term as the last column before the labels
        self.training_data.insert(len(self.training_data.columns) - 1, "bias", 1)
        self.do_objective_output = do_objective_output

        if gammaFunc == 'A':
            self.gammaFunc = self.gammaA
        else:
            self.gammaFunc = self.gammaB

        self.output_column = output_column
        self.C = C
        self.w: np.ndarray = np.zeros(len(self.training_data.columns) - 1) 
        self.epochs = 0

        x = self.training_data.drop([output_column], axis=1)
        y = self.training_data[output_column]
        self.outputs = []

        for i in range(epochs):
            self.add_epoch()
            #print(str(self.epochs), end=" ", flush=True)

            if self.do_objective_output:

                sum = 0.0
                for xi,yi in zip(x, y):
                    temp = 1 - yi * self.w.T.dot(xi)
                    if temp > 0:
                        sum = sum + temp

                sum = C * sum
                sum = sum + 0.5 * self.w[0:-1].T.dot(self.w[0:-1])
                self.outputs.append(sum)


        
        print(self.w)
        print()

    def get_label(self, row: Series):
        # row needs to be augmented to support the bias.
        rowArr = row.drop(self.output_column)
        rowArr = np.append(rowArr, 1)

        output = self.w.T.dot(rowArr)

        if output >= 0:
            return 1
        else:
            return -1