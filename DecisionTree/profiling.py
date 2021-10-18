import cProfile

from typing import List
from pandas.core.frame import DataFrame
from decisionTree import DecisionTree, getMajorityError, getGiniIndex, getEntropy, node
import pandas as df
#import threading as th
import multiprocessing as pr
from ete3 import Tree, TreeStyle, TextFace, add_face_to_node

car_frame_train = df.read_csv("car/train.csv", names=["buying", "maint", "doors", "persons", "lug_boot", "safety", "label"])
car_frame_test = df.read_csv("car/test.csv", names=["buying", "maint", "doors", "persons", "lug_boot", "safety", "label"])

bank_frame_train = df.read_csv("bank/train.csv", names=["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", \
    "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"])

bank_frame_test = df.read_csv("bank/test.csv", names=["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", \
    "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"])

#Convert these data frames to numeric
bank_frame_train.apply(df.to_numeric, errors="ignore")
bank_frame_test.apply(df.to_numeric, errors="ignore")

def printChildren(aNode: node, tree, attribute: str = " "):
    if aNode.is_leaf:
        treeChild = tree.add_child(name=(attribute + " " + aNode.label))
        return
    else:
        treeChild = tree.add_child(name=aNode.best_attribute)
        nextAttrib =  aNode.best_attribute


    print(str(len(aNode.children)) + " ", end=" ")
    for child in aNode.children:
        printChildren(aNode.children[child], treeChild, child + " " + nextAttrib)

bank_schema = \
{ "age":[], \
    "job":["admin.","unknown","unemployed","management","housemaid","entrepreneur","student", "blue-collar","self-employed","retired","technician","services" ] ,\
    "marital":[ "married","divorced","single"] ,\
    "education":[ "unknown","secondary","primary","tertiary"] ,\
    "default":[  "yes","no"] ,\
    "balance":[ ] ,\
    "housing":[ "yes","no"] ,\
    "loan":[ "yes","no"] ,\
    "contact":[  "unknown","telephone","cellular"] ,\
    "day":[ ] ,\
    "month":[ "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"] ,\
    "duration":[ ] ,\
    "campaign":[ ] ,\
    "pdays":[ ] ,\
    "previous":[ ] ,\
    "poutcome":["unknown","other","failure","success"], \
    "y": ["yes", "no"]}

car_schema = {'buying':['vhigh', 'high', 'med', 'low'], \
    'maint':['vhigh', 'high', 'med', 'low'], \
    'doors':['2', '3', '4','5more'], \
    'persons':['2', '4', 'more'], \
    'lug_boot':['small', 'med', 'big'], \
    'safety':['low', 'med', 'high'], \
    'label':['unacc', 'acc', 'good', 'vgood']}

tennis_schema = {'O': ['S','O','R'], 'T': ['H', 'M', 'C'], 'H': ['H', 'N', 'L'], 'W': ['S', 'W'], 'play': ['0', '1']}

print("building tree")
dt = DecisionTree(bank_frame_train, "y", bank_schema, 2, getEntropy)
print("making ete tree")
t = Tree(format=8)
printChildren(dt.root, t)

def my_layout(node):
        F = TextFace(node.name, tight_text=True)
        add_face_to_node(F, node, column=0, position="branch-right")

#print(t.get_ascii(show_internal=True))
ts = TreeStyle()
ts.show_leaf_name = False
def my_layout(node):
        F = TextFace(node.name, tight_text=True)
        add_face_to_node(F, node, column=0, position="branch-right")
ts.layout_fn = my_layout
t.show(tree_style=ts)
exit()

cProfile.run('DecisionTree(bank_frame_train, "y", bank_schema, 17, getEntropy)')