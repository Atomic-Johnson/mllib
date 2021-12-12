### This is a machine learning library developed by Ted Goodell forCS5350/6350 in University of Utah.

<h1> decision tree </h1>

The decision tree is implemented in the folder decision tree. It contains a file called DT_tester.py
DT_tester.py will evaluate the performance of the decision tree that I wrote which is contained in
decisionTree.py. Specifically it will evaluate the perfomance on six data sets: car test and training
bank test and training, and bank test and training (with 'unknown' values replaced) in that order.

THIS CODE IS TIME CONSUMING! Given the size of this dataset, the DT_tester script can take at least
20 minutes to run.

This decision tree algorithm comes with no warranty implicit or otherwise and should not be relied upon to make
important deicions regarding your career, dating life, or education.

Here is the docstring and method for the decision tree constructor:  

'''python
def __init__(self, training_data: DataFrame, label_column:str, schema: dict = None, max_depth: Number = 90, purity_func: Callable[[DataFrame, str, bool], float] = getEntropy):
        Initialize A decision tree object. 
'''

### Args:
* training_data (DataFrame): The data that will be used to make this tree. Must have header that matches keys in schema. Numeric columns must have numeric tyoe data.
* label_column (str): The column in training_data that has the output labels
* schema (dict, optional): a dictionary where each key is a column in training_data and corresponds to a list of possible values for that column. Defaults to None.
* max_depth (Number, optional): the maximum depth of this tree. Defaults to 90.
* purity_func (Callable[[DataFrame, str], float], optional): The function that will be used to calculate purity for the information gain (Use getEntropy, getMajorityError, or getGiniIndex contained in decisionTree.py). Defaults to getEntropy.
        
once the tree is constructed, you can get a prediction from it by calling getLabel
```python
def getLabel(self, test_data: dict):
#Get the predicted label from the decision tree for test_data. This recursively calls get_label on this tree's root node
```

### Args:
  * test_data (dict): a dictionary corresponding to a single row in a dataset. The keys must be the same as the column names in the training_data DataFrame for this tree. To get such a dictionary from a DataFrame, you can use DataFrame.to_dict()

Returns: The predicted label
        
# Adaboost, Bagging, and Random Forest

The code to generate the outputs for this portion on the assignment can be executed simply by running run.sh from the console. run.sh does not expect any command line args but you do need to be in the Ensemble Learning directory to run the code.  

## Adaboost
The Adaboost library is contained in adaboost.py. It's interface is essentially the same as the interface for the decision tree library except that the constructor has a parameter T. and there is a function to add an iteration. Information gain is calculated useing entropy.

```python
def __init__(self, training_data: DataFrame, T, output_column: str, schema: dict):
```
### Args:
* training_data (DataFrame): The data that will be used to make this tree. Must have header that matches keys in schema. Numeric columns must have numeric tyoe data.
* label_column (str): The column in training_data that has the output labels
* T: The initial number of iterations to compute
* schema (dict, optional): a dictionary where each key is a column in training_data and corresponds to a list of possible values for that column. Defaults to None.

## Bagging
The bagging library is in treeBaggerMultiProc.py which has been enhanced to use multiprocessing. The constructor and the getLabel function for the BaggedTree class takes the same parameters as the AdaBoost library. This library also includes the following function which gets the label for each iteration of the bagging class.

```python
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
```

## Random Forest
The random forest is contained in randomForest.py. It contains an altered version of the decision tree library to allow for random feature selection and the RandomForest class. This class has a constructor with the same parameteres as the bagging library with the exception that it has a parameter to set how many random features to select.  

```python
def __init__(self, training_data: DataFrame, T, feature_sample_size:int, output_column: str, schema: dict):
```

 It also contains a getLabelsUpToT function like the bagging library above
 
 # Perceptron Algorithms
 These algorithms can be tested by running the run.sh file in the Perceptron directory. That script must be run from the Perceptron directory. There are three classes: Averaged_Perceptron, Perceptron, and Voted_Perceptron and they exist in similarly named python files. They each have identical interfaces to their constructors and other methods as listed below.
 
 ```python
 def __init__(self, training_data: DataFrame, output_column: str, epochs, r = 0.5) -> None:
 def get_label(self, row: Series):
 def add_epoch(self):
 ```
 
 The add_epoch function allow you manually add epochs to the perceptron and check the performance of the model at each epoch. However, if you do this, you should specify epochs=0 inyour constructor. This is how I evaluate the performance of the different perceptrons at different epochs in perceptron_results.py which is run from run.sh
