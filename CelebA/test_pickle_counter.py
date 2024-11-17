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

curr_req = []
counter_1 = 0
counter_2 = 0
spurious_counter = 0
for i, j, k in d:
    if k == 1:
        if counter_1 < 5000*0.1:
            if int(d1.loc[i, 'Eyeglasses']) == 1:
                spurious_counter += 1
            curr_req.append(i)
            counter_1 += 1
    else:
        if counter_2 < 5000*0.1:
            curr_req.append(i)
            counter_2 += 1

print(len(curr_req))
print(5000 - 5000*0.1, 2500 - spurious_counter)


curr_req = []
counter_1 = 0
counter_2 = 0
spurious_counter = 0
for i, j, k in d:
    if k == 1:
        if counter_1 < 5000*0.25:
            if int(d1.loc[i, 'Eyeglasses']) == 1:
                spurious_counter += 1
            curr_req.append(i)
            counter_1 += 1
    else:
        if counter_2 < 5000*0.25:
            curr_req.append(i)
            counter_2 += 1

print(len(curr_req))
print(5000 - 5000*0.25, 2500 - spurious_counter)

curr_req = []
counter_1 = 0
counter_2 = 0
spurious_counter = 0
for i, j, k in d:
    if k == 1:
        if counter_1 < 5000*0.4:
            if int(d1.loc[i, 'Eyeglasses']) == 1:
                spurious_counter += 1
            curr_req.append(i)
            counter_1 += 1
    else:
        if counter_2 < 5000*0.4:
            curr_req.append(i)
            counter_2 += 1

print(len(curr_req))
print(5000 - 5000*0.4, 2500 - spurious_counter)


curr_req = []
counter_1 = 0
counter_2 = 0
spurious_counter = 0
for i, j, k in d:
    if k == 1:
        if counter_1 < 5000*0.5:
            if int(d1.loc[i, 'Eyeglasses']) == 1:
                spurious_counter += 1
            curr_req.append(i)
            counter_1 += 1
    else:
        if counter_2 < 5000*0.5:
            curr_req.append(i)
            counter_2 += 1

print(len(curr_req))
print(5000 - 5000*0.5, 2500 - spurious_counter)


curr_req = []
counter_1 = 0
counter_2 = 0
spurious_counter = 0
for i, j, k in d:
    if k == 1:
        if counter_1 < 5000*0.75:
            if int(d1.loc[i, 'Eyeglasses']) == 1:
                spurious_counter += 1
            curr_req.append(i)
            counter_1 += 1
    else:
        if counter_2 < 5000*0.75:
            curr_req.append(i)
            counter_2 += 1

print(len(curr_req))
print(5000 - 5000*0.75, 2500 - spurious_counter)


curr_req = []
counter_1 = 0
counter_2 = 0
spurious_counter = 0
for i, j, k in d:
    if k == 1:
        if counter_1 < 5000*0.9:
            if int(d1.loc[i, 'Eyeglasses']) == 1:
                spurious_counter += 1
            curr_req.append(i)
            counter_1 += 1
    else:
        if counter_2 < 5000*0.9:
            curr_req.append(i)
            counter_2 += 1

print(len(curr_req))
print(5000 - 5000*0.9, 2500 - spurious_counter)


curr_req = []
counter_1 = 0
counter_2 = 0
spurious_counter = 0
for i, j, k in d:
    if k == 1:
        if counter_1 < 5000*0.95:
            if int(d1.loc[i, 'Eyeglasses']) == 1:
                spurious_counter += 1
            curr_req.append(i)
            counter_1 += 1
    else:
        if counter_2 < 5000*0.95:
            curr_req.append(i)
            counter_2 += 1

print(len(curr_req))
print(5000 - 5000*0.95, 2500 - spurious_counter)


curr_req = []
counter_1 = 0
counter_2 = 0
spurious_counter = 0
for i, j, k in d:
    if k == 1:
        if counter_1 < 5000*0.97:
            if int(d1.loc[i, 'Eyeglasses']) == 1:
                spurious_counter += 1
            curr_req.append(i)
            counter_1 += 1
    else:
        if counter_2 < 5000*0.97:
            curr_req.append(i)
            counter_2 += 1

print(len(curr_req))
print(5000 - 5000*0.97, 2500 - spurious_counter)

