import numpy as np
from scipy.optimize import minimize
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from scipy.optimize import OptimizeResult

class DualSVM():
    def gaussKernel(self, xi: np.ndarray):
        xi = np.asmatrix(xi)
        return np.exp(-((np.sum(np.multiply(xi, xi), axis = 1) + np.sum(np.multiply(xi.T, xi.T) , axis = 0) - 2 * xi.T * xi) / self.gamma))
        #return np.exp(-((np.sum(xi.T.dot(xi), axis = 1) + np.sum(np.multiply(xi.T, xi.T) , axis = 0) + 2 * xi.T @ xi) / self.gamma))

    def gaussKernelDual(self, xi: np.ndarray, xj: np.ndarray):
        return np.exp(- (((xi-xj)**2).sum())/self.gamma)

    def gaussObjective(self, alphas: np.ndarray):
        #return 0.5 * (self.gaussKernel(self.x) *  np.dot( (alphas * self.y).T, (alphas*self.y) )  ).sum() - alphas.sum()
        return 0.5 * (self.xg *  np.dot((alphas * self.y).T, (alphas*self.y))  ).sum() - alphas.sum()

    def constraints(self, alphas: np.ndarray):
        return (alphas * self.yarr).sum()

    def objective(self,alphas: np.ndarray):
        return 0.5 * (np.dot(self.x, self.x.T) *  np.dot((alphas * self.y).T, (alphas*self.y))  ).sum() - alphas.sum()

    def __init__(self, training_data: DataFrame, output_column: str, C = 0.5, useGaussKernel=False, gamma = 1) -> None:
        self.training_data = training_data.copy(deep=True)
        #Let's wrap in a bias term as the last column before the labels
        self.training_data.insert(len(self.training_data.columns) - 1, "bias", 1)

        self.y = np.array([self.training_data[output_column].to_numpy()])
        self.yarr = self.training_data[output_column].to_numpy()
        self.gamma = gamma
        self.useGaussKernel = useGaussKernel
        
        self.yarr = self.training_data[output_column].to_numpy()
        self.x = self.training_data.drop([output_column], axis=1).to_numpy()
        self.alphas = np.random.rand(len(self.x))
        self.bounds = [(0,C)] * len(self.alphas)

        self.output_column = output_column
        self.C = C

        cons = ({'type':'eq', 'fun':self.constraints})
        if useGaussKernel:
            print("building kernel")
            self.xg = np.zeros((len(self.x), len(self.x)))
            for i in range(len(self.x)):
                xi = self.x[i]
                for j in range(len(self.x)):
                    self.xg[i][j] = self.gaussKernelDual(xi, self.x[j])

            #self.xg = self.x.T.dot(np.sum(self.x.dot(self.x), axis=1))
            print("optimizing")
            solution: OptimizeResult = minimize(fun=self.gaussObjective, x0=self.alphas, constraints=cons, bounds=self.bounds, method='SLSQP')
        else:
            solution: OptimizeResult = minimize(fun=self.objective, x0=self.alphas, constraints=cons, bounds=self.bounds, method='SLSQP')

        self.alphas = solution.x
        print(solution.message)
        self.w: np.ndarray = np.zeros(len(self.training_data.columns) - 1) 
        
        # this part is slow but we only need to do it once!
        if not useGaussKernel:
            for a,yi,x in zip(self.alphas, self.training_data[output_column].to_numpy(), self.x):
                if a == 0:
                    continue
                self.w = self.w + (a * yi) * x

        #self.w = (self.alphas * self.y * self.x).sum(axis=1)
      


    def get_label(self, row: Series):
        # row needs to be augmented to support the bias.
        rowArr = row.drop(self.output_column).to_numpy()
        rowArr = np.append(rowArr, 1)
        

        if self.useGaussKernel:
            #rowArr = np.asmatrix(rowArr)
            prediction = 0.0
            for i in range(len(self.alphas)):
                if self.alphas[i] == 0: 
                    continue
                prediction = prediction + self.alphas[i] * self.yarr[i] * self.C * self.gaussKernelDual(self.x[i], rowArr)

            output = prediction
        else:
            output = self.w.T.dot(rowArr)

        if output >= 0:
            return 1
        else:
            return -1