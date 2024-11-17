import pickle
from collections import Counter
import pandas as pd

d1 = pd.read_csv('celebA/data/metadata_main.csv')

with open('9el2n.pkl', 'rb') as f:
    d = pickle.load(f)

print(len(d))

x = []
for i, j, k in d:
    x.append(i)

d.sort(key = lambda x:-x[1])
d_new = [i for i in d if i[2] == 1]

spurious_counter = 0
nsp_counter = 0

for ind, (i, j, k) in enumerate(d_new):
    if ind in (1250, 2500, 3750):
        print(spurious_counter, nsp_counter)
        spurious_counter = 0
        nsp_counter = 0
    if int(d1.loc[i, 'Eyeglasses']) == 1:
        spurious_counter += 1
    else:
        nsp_counter += 1

print(spurious_counter, nsp_counter)
