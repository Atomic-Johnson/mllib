import pandas as df
import unittest
from perceptron import Perceptron

class PerceptronTests(unittest.TestCase):
    def setUp(self):
        self.data = df.DataFrame({'x1':[0, -1, 0, 1], 'x2':[1, 0, -1, 0], 'y':[1, -1, -1, 1]})

    def testWeights(self):
        percy = Perceptron(self.data,'y', 10)
        print(percy.w)