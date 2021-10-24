import pandas as df
import numpy as np
from lms import lms
import matplotlib.pyplot as plt

column_names = [ "Cement", "Slag", "Fly ash", "Water", "SP", "Coarse Aggr", "Fine Aggr", "output"]

concrete_train = df.read_csv("concrete/train.csv", names=column_names)
concrete_test = df.read_csv("concrete/test.csv", names=column_names)

stochastic_model = lms(concrete_train, "output", True, iterations=2500, rate=0.006, adjustRate=False)
print("finished stochastic model")


#for stochastic descent we need final weight vector, learning rate, figure of cost at each step
print("gradient; weight:" + str(stochastic_model.w))
print(" final rate:" + str(stochastic_model.rate))
print("cost: " + str(stochastic_model._get_cost()))
plt.plot(stochastic_model.costs)
plt.title("stochasitic: cost vs iteration")
plt.ylabel("cost")
plt.xlabel("iterations")
plt.show()

gradient_model = lms(concrete_train, "output", False, iterations=800)
print("finished gradient model")

#for gradient descent we need final weight vector, learning rate, figure of cost at each step
print("gradient; weight:" + str(gradient_model.w))
print(" final rate:" + str(gradient_model.rate))
print("cost: " + str(gradient_model._get_cost()))
plt.plot(gradient_model.costs)
plt.title("gradient: cost vs iteration")
plt.ylabel("cost")
plt.xlabel("iterations")
plt.show()
