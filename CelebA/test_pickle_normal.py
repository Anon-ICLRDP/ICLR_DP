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

d.sort(key = lambda x:x[1])

curr_req = []
counter_1 = 0
counter_2 = 0
for i, j, k in d:
    if k == 1:
        if counter_1 < 5000*0.1:
            curr_req.append(i)
            counter_1 += 1
    else:
        if counter_2 < 5000*0.1:
            curr_req.append(i)
            counter_2 += 1

print(len(curr_req))

with open('10_percent_n.pkl', 'wb') as f:
    pickle.dump(curr_req, f)


curr_req = []
counter_1 = 0
counter_2 = 0
for i, j, k in d:
    if k == 1:
        if counter_1 < 5000*0.25:
            curr_req.append(i)
            counter_1 += 1
    else:
        if counter_2 < 5000*0.25:
            curr_req.append(i)
            counter_2 += 1

print(len(curr_req))

with open('25_percent_n.pkl', 'wb') as f:
    pickle.dump(curr_req, f)

curr_req = []
counter_1 = 0
counter_2 = 0
for i, j, k in d:
    if k == 1:
        if counter_1 < 5000*0.4:
            curr_req.append(i)
            counter_1 += 1
    else:
        if counter_2 < 5000*0.4:
            curr_req.append(i)
            counter_2 += 1

print(len(curr_req))

with open('40_percent_n.pkl', 'wb') as f:
    pickle.dump(curr_req, f)

# 50% of the samples with class = male

curr_req = []
counter_1 = 0
counter_2 = 0
for i, j, k in d:
    if k == 1:
        if counter_1 < 5000*0.5:
            curr_req.append(i)
            counter_1 += 1
    else:
        if counter_2 < 5000*0.5:
            curr_req.append(i)
            counter_2 += 1

print(len(curr_req))

with open('50_percent_n.pkl', 'wb') as f:
    pickle.dump(curr_req, f)

curr_req = []
counter_1 = 0
counter_2 = 0
for i, j, k in d:
    if k == 1:
        if counter_1 < 5000*0.75:
            curr_req.append(i)
            counter_1 += 1
    else:
        if counter_2 < 5000*0.75:
            curr_req.append(i)
            counter_2 += 1

print(len(curr_req))

with open('75_percent_n.pkl', 'wb') as f:
    pickle.dump(curr_req, f)

curr_req = []
counter_1 = 0
counter_2 = 0
for i, j, k in d:
    if k == 1:
        if counter_1 < 5000*0.9:
            curr_req.append(i)
            counter_1 += 1
    else:
        if counter_2 < 5000*0.9:
            curr_req.append(i)
            counter_2 += 1

print(len(curr_req))

with open('90_percent_n.pkl', 'wb') as f:
    pickle.dump(curr_req, f)

curr_req = []
counter_1 = 0
counter_2 = 0
for i, j, k in d:
    if k == 1:
        if counter_1 < 5000*0.95:
            curr_req.append(i)
            counter_1 += 1
    else:
        if counter_2 < 5000*0.95:
            curr_req.append(i)
            counter_2 += 1

print(len(curr_req))

with open('95_percent_n.pkl', 'wb') as f:
    pickle.dump(curr_req, f)

curr_req = []
counter_1 = 0
counter_2 = 0
for i, j, k in d:
    if k == 1:
        if counter_1 < 5000*0.97:
            curr_req.append(i)
            counter_1 += 1
    else:
        if counter_2 < 5000*0.97:
            curr_req.append(i)
            counter_2 += 1

print(len(curr_req))

with open('97_percent_n.pkl', 'wb') as f:
    pickle.dump(curr_req, f)
