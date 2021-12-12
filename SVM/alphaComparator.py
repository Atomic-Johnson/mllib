import pandas as pd

df0_1 = pd.read_csv("0.1.csv")
df0_5 = pd.read_csv("0.5.csv")
df1 = pd.read_csv("1.0.csv")
df5 = pd.read_csv("5.0.csv")
df100 = pd.read_csv("100.0.csv")

sum = 0
for a1,a2 in zip(df0_1.iterrows(), df0_5.iterrows()):
    if a1[1][0] == a2[1][0]:
        sum = sum + 1

print("equal alphas between 0.1 and 0.5 = " + str(sum))

sum = 0
for a1,a2 in zip(df0_5.iterrows(), df1.iterrows()):
    if a1[1][0] == a2[1][0]:
        sum = sum + 1

print("equal alphas between 0.5 and 1 = " + str(sum))

sum = 0
for a1,a2 in zip(df1.iterrows(), df5.iterrows()):
    if a1[1][0] == a2[1][0]:
        sum = sum + 1

print("equal alphas between 1 and 5 = " + str(sum))

sum = 0
for a1,a2 in zip(df5.iterrows(), df100.iterrows()):
    if a1[1][0] == a2[1][0]:
        sum = sum + 1

print("equal alphas between 5 and 100 = " + str(sum))