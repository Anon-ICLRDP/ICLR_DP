import pickle
import pandas as pd
import numpy as np

d = pd.read_csv('logs//test.csv')

for i in range(d.shape[0]):
    print(i+1, d.loc[i, 'avg_acc_group:0'], d.loc[i, 'avg_acc_group:1'], d.loc[i, 'avg_acc_group:2'], d.loc[i, 'avg_acc_group:3'], d.loc[i, 'avg_acc_group:4'], d.loc[i, 'avg_acc_group:5'])
print()
d = pd.read_csv('logs//val.csv')
minepoch = 0
m = -float('inf')
for i in range(d.shape[0]):
    #print(i+1, d.loc[i, 'avg_acc_group:0'], d.loc[i, 'avg_acc_group:1'], d.loc[i, 'avg_acc_group:2'], d.loc[i, 'avg_acc_group:3'], d.loc[i, 'avg_acc_group:4'], d.loc[i, 'avg_acc_group:5'], min(d.loc[i, 'avg_acc_group:0'], d.loc[i, 'avg_acc_group:1'], d.loc[i, 'avg_acc_group:2'], d.loc[i, 'avg_acc_group:3'], d.loc[i, 'avg_acc_group:4'], d.loc[i, 'avg_acc_group:5']))
    print(i+1, d.loc[i, 'avg_acc_group:0'], d.loc[i, 'avg_acc_group:1'], d.loc[i, 'avg_acc_group:2'], d.loc[i, 'avg_acc_group:3'], d.loc[i, 'avg_acc_group:4'], d.loc[i, 'avg_acc_group:5'])
    if d.loc[i, 'avg_acc_group:5'] > m:
        m = d.loc[i, 'avg_acc_group:5']
        minepoch = i + 1

print(minepoch)
