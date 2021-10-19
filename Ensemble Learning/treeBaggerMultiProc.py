from typing import List, Type
from decisionTree import DecisionTree
import pandas as df
import math as m
import numpy as np
import multiprocessing as mp
from pandas.core.frame import DataFrame

def _buildATree(training_data: DataFrame, output_column: str, schema: dict, queue:mp.Queue):
    tempTree = DecisionTree(training_data, output_column, schema)
    queue.put(tempTree, block=True, timeout=3)
    queue.close()
    print("added a tree")

class BaggedTree:

    def add_iterations(self, T):
        lock = mp.Lock()
        procs = []
        treeQueue = mp.Queue(maxsize=75) 
        for i in range(0,T):
            treeBuilder = mp.Process(target=_buildATree, \
                args=(self.training_data.sample(self.sample_size, replace=True), self.output_column, self.schema, treeQueue))
            
            procs.append(treeBuilder)
            treeBuilder.start()
            if not treeQueue.empty():
                self.trees.append(treeQueue.get())
                
            while len(procs) >= mp.cpu_count():
                oldest_proc = procs.pop(0)
                oldest_proc.join()

        # Wait for remaining trees to finish building
        for proc in procs:
            proc.join()

        while not treeQueue.empty():
            self.trees.append(treeQueue.get())

        self.T = self.T + T
        print(len(self.trees))
                

    def add_iteration(self):
        bootstrap = self.training_data.sample(self.sample_size, replace=True)
        self.trees.append(DecisionTree(bootstrap, self.output_column, self.schema))
        self.T = self.T + 1
        
    def __init__(self, training_data: DataFrame, T, output_column: str, schema: dict):
        self.T = 0
        self.sample_size = len(training_data)
        self.output_column = output_column
        self.schema = schema
        self.training_data: DataFrame = training_data
        self.trees:List[DecisionTree] = []

        self.add_iterations(T)

    def getLabel(self, test_data: dict):
        sum = 0
        for tree in self.trees:
            sum = sum + tree.getLabel(test_data)

        if sum >= 0:
            return 1
        else:
            return -1

    def getLabelsUpToT(self, test_data:dict, T:int):
        sum = 0
        labels = []
        for i in range(0,T):
            sum = sum + self.trees[i].getLabel(test_data)

            if sum >= 0:
                labels.append(1)
            else:
                labels.append(-1)

        return labels