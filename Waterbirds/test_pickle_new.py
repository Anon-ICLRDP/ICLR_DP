import pickle
import torch

with open('90/19el2n.pkl', 'rb') as f:
    x = pickle.load(f)
x.sort(key = lambda x:-x[1])
ce, cn = 0, 0
for idx, (i, j, k, l) in enumerate(x):
    if idx < (len(x))//5:
        if k == l:
            ce += 1
        else:
            cn += 1

#print((ce/4555)/(cn/240))
#print((ce/(1761 + 670))/(cn/(181 + 55)))
#print((ce/(898 + 476))/(cn/(174 + 55)))
print((ce/(385 + 360))/(cn/(165 + 55)))
print(ce, cn)
