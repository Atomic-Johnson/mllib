import numpy as np
import pandas as pd
import math as m
from pandas.core.series import Series

from pandas.core.frame import DataFrame

class ann:
    def gammat(self,t):
        gamma0 = 0.07
        d = 0.75
        return gamma0 / (1 + gamma0/d * t)

    def sigmoid(self, s):
        return 1/(1 + np.exp(-s))

    def do_back_prop(self, example: Series):

        dldz_2 = np.zeros(self.L2Size - 1)
        dldz_1 = np.zeros(self.L2Size - 1)

        #make set 1
        dLdy = self.get_prediction(example) - example[self.output_col]
        self.dldw_2out = dLdy * self.z2

        #make set 2
        dldz_2 = dLdy * self.w2out[1:] #because we need to ignore the bias vector

        # for i in range(self.L1Size):
        #     self.dldw_12[i] = dldz_2 * self.z2[1:] * (1 - self.z2[1:]) * self.z1[i]
        weird_Dz_vector = dldz_2 * self.z2[1:] * (1 - self.z2[1:])
        self.dldw_12 = np.asmatrix(self.z1).T.dot(np.asmatrix(weird_Dz_vector))

        #make set 3
        weird_Dz_vector = dldz_2 * self.z2[1:] * (1 - self.z2[1:])
        dldz_1 = np.asarray(self.w12[1:].dot(weird_Dz_vector))

        weird_Dz_vector = dldz_1 * self.z1[1:] * (1 - self.z1[1:])
        exarr = example.drop(self.output_col).to_numpy()
        exarr:np.ndarray = np.insert(exarr,0, 1) #add bias

        self.dldw_01 = np.asmatrix(exarr).T.dot(np.asmatrix(weird_Dz_vector))
        # for i in range(self.L1Size):
        #     #compensate for bias
        #     if i > 0:
        #         self.dldw_01[i] = weird_Dz_vector * example[i-1]
        #     else:
        #         self.dldw_01[i] = weird_Dz_vector * 1
        

    def __init__(self, training_data: DataFrame, output_column:str, iterations = 500, randomW = True, L1size = 3, L2size = 3) -> None:
        self.training_data: DataFrame = training_data.copy(deep=True)
        #self.training_data.insert(0, 'bias', 1) #add ones for the bias calculation
        self.train_data_noY = self.training_data.drop(output_column, axis=1)
        self.output_col = output_column

        self.L2Size = L2size
        self.L1Size = L1size
        
        self.dldw_2out = np.zeros(self.L2Size)
        self.dldw_12 = np.zeros((self.L1Size, self.L2Size -1))
        self.dldw_01 = np.zeros((self.train_data_noY.shape[1]+1, self.L1Size - 1))

        rs = np.random.RandomState(5)

        #each row corresponds to the weights from 1 previous node
        if randomW:
            self.w01 = rs.normal(size=(self.train_data_noY.shape[1]+1, L1size - 1))
            self.w12 = rs.normal(size=(L1size, L2size - 1))
            self.w2out = rs.normal(size=L2size)
        else:
            self.w01 = np.zeros((self.train_data_noY.shape[1]+1, L1size - 1))
            self.w12 = np.zeros((L1size, L2size - 1))
            self.w2out = np.zeros(L2size)

        self.z1 = np.zeros(L1size)
        self.z1[0] = 1
        self.z2 = np.zeros(L2size)
        self.z2[0] = 1

        self.sigmoid_v = np.vectorize(self.sigmoid)

        for i in range(iterations):
            self.do_SGD(i)

    def get_prediction(self, row: Series):
        #remove output and add bias
        rowList = row.drop(self.output_col).to_list()
        rowList.insert(0, 1)
        rowArr = np.array(rowList)

        #reinitialize z values
        self.z1.fill(0)
        self.z1[0] = 1
        self.z2.fill(0)
        self.z2[0] = 1

        for x, w01x in zip(rowArr, self.w01):
            self.z1[1:] = self.z1[1:] + x * w01x

        self.z1[1:] = self.sigmoid_v(self.z1[1:])

        for z1x, w12x in zip(self.z1, self.w12):
            self.z2[1:] = self.z2[1:] + z1x * w12x

        self.z2[1:] = self.sigmoid_v(self.z2[1:])

        output = self.z2.T.dot(self.w2out)
        return output

    def do_SGD(self, t):
        shuffle_data = self.training_data.sample(len(self.training_data), replace=False)

        for i,row in shuffle_data.iterrows():
            self.do_back_prop(row)
            gamma_t = self.gammat(t)

            self.w01 = self.w01 - gamma_t * self.dldw_01
            self.w12 = self.w12 - gamma_t * self.dldw_12
            self.w2out = self.w2out - gamma_t * self.dldw_2out

        
                



        
