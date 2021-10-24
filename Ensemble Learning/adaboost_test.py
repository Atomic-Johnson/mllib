from adaboost import boostedTree
import pandas as df
import unittest

class adaBoostTest(unittest.TestCase):
    def setUp(self):
        self.tennis_schema = {'O': ['S','O','R'], 'T': ['H', 'M', 'C'], 'H': ['H', 'N', 'L'], 'W': ['S', 'W'], 'play': ['0', '1']}
        self.testData = df.read_csv("basicTest.csv", names=['O', 'T', 'H', 'W', 'play'])
        self.testData.convert_dtypes()
        self.boostedtree = boostedTree(self.testData,0,"play", self.tennis_schema)

    def test_addIterations(self):
        self.assertListEqual(self.testData["weights"].to_list(),[1/14]*14)
        self.boostedtree.add_iteration()
        neww = [0.05, \
                0.05, \
                0.05, \
                0.05, \
                0.05, \
                0.124999954, \
                0.05, \
                0.05, \
                0.124999954, \
                0.05, \
                0.124999954, \
                0.05, \
                0.05, \
                0.124999954
            ]

        for w, excelw in zip(self.testData["weights"].to_list(), neww):
            self.assertAlmostEqual(w, excelw, 6)
        
        self.boostedtree.add_iteration()
        self.boostedtree.add_iteration()
        self.boostedtree.add_iteration()
        self.boostedtree.add_iteration()

if __name__ == '__main__':
    unittest.main()