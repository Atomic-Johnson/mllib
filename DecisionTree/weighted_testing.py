from decisionTree import DecisionTree
import pandas as df


tennisDF = df.read_csv("tennis-missing.csv")
tennisDF.apply(df.to_numeric, errors="ignore")

print(tennisDF)


tennisSchema = {"O":["S","O","R"], "T":["H","M","C"], "H":["H","N","L"], "W":["S", "W"]}
tennisTree = DecisionTree(tennisDF, "play", tennisSchema)

print(tennisTree)