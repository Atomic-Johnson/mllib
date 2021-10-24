import pandas as df
from pandas.core.frame import DataFrame
import numpy as np
from numpy.linalg import norm
from typing import Callable

def get_gradient_descent(w: np.ndarray, data: DataFrame, output_column: str, rate) -> np.ndarray:
    sum = 0
    gradient = np.zeros(len(w))
   
    for j in range(0, len(w)):

        sum = 0
        for i, r in data.iterrows():
            xi = np.array(r[0 : len(w)]) #get the numpy array of the inputs
            sum_iteration = float(r[output_column]) - w.T.dot(xi) 
            sum = sum + sum_iteration * r[j] # multiply by xij
        
        gradient[j] = -1 * sum

    new_w = w - rate * gradient
    return new_w

def get_stochastic_gradient(w: np.ndarray, data_row: DataFrame, output_column: str, r) -> np.ndarray:
    xi = np.array(data_row[data_row.columns[0 : len(w)]].values.flatten()) #get the numpy array of the inputs
    new_w = np.zeros(len(w))

    for j in range(0,len(w)):
        new_w[j] = w[j] + r*(data_row[output_column] - w.T.dot(xi)) * xi[j]

    return new_w



class lms:
    def _get_cost(self):
        costs = np.zeros(len(self.training_data))

        for i, row in self.train_data_noY.iterrows():
            rownp = row.to_numpy()
            costs[i] = (self.training_data[self.output_column].iat[i] - self.w.T.dot(rownp))**2

        return costs.sum() / 2

    def __init__(self, training_data: DataFrame, output_column:str, do_stochastic_descent: bool, iterations = 500, rate = 1, adjustRate: bool = True) -> None:
        
        self.training_data: DataFrame = training_data.copy(deep=True)
        self.training_data.insert(0, 'bias', 1) #add ones for the bias calculation
        self.train_data_noY = self.training_data.drop(output_column, axis=1)
        self.w = np.zeros(len(self.train_data_noY.columns))
        self.iterations = iterations
        self.rate = rate
        self.tolerance = 1e-2
        self.output_column = output_column
        self.costs = np.zeros(self.iterations)

        if do_stochastic_descent:
            for i in range(self.iterations):
                sample_row = self.training_data.sample(1)
                new_w = get_stochastic_gradient(self.w, sample_row, self.output_column, self.rate)

                while norm(new_w - self.w) > self.tolerance and adjustRate: # does it converge?
                    self.rate = self.rate / 2
                    new_w = get_stochastic_gradient(self.w, sample_row, self.output_column, self.rate)
                    
                self.costs[i] = self._get_cost()
                self.w = new_w
        else:
            for i in range(self.iterations):
                new_w = get_gradient_descent(self.w, self.training_data, self.output_column, self.rate)

                while norm(new_w - self.w) > self.tolerance and adjustRate: # does it converge?
                    self.rate = self.rate / 2
                    new_w = get_gradient_descent(self.w, self.training_data, self.output_column, self.rate)

                self.costs[i] = self._get_cost()
                self.w = new_w

    def get_label(self, row: dict, output_column: str):
        row = row.pop(output_column)
        nprow = np.array(row.values())

        return self.w.T.dot(nprow)
