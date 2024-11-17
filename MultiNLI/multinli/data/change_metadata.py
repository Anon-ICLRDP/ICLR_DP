import pickle
import pandas as pd
import os

from tqdm import tqdm

with open('../../4el2n_cheap.pkl', 'rb') as f:
    x = pickle.load(f)
'''
new_one = []

for i, j, k, l in x:
    new_one.append([i, j.item(), k, l])

with open('../../4el2n_cheap.pkl', 'wb') as f:
    pickle.dump(new_one, f)
exit()
'''
print('s1')
x.sort(key = lambda x:-x[1])
new_0 = []
new_1 = []
new_2 = []
new_3 = []
for i, j, k, l in x:
    if k == 0 and l == 0:
        new_0.append(i)
    if k == 0 and l == 1:
        new_3.append(i)
    elif k == 1 and l == 2:
        new_1.append(i)
    elif k == 2 and l == 4:
        new_2.append(i)

print(len(new_1), len(new_2), len(new_0), len(new_3))

new_0 = new_0[:int(len(new_0)//5)]
new_0 = set(new_0)

new_1 = new_1[:int(len(new_1)//5)]
new_1 = set(new_1)

new_2 = new_2[:int(len(new_2)//5)]
new_2 = set(new_2)

new_3 = new_3[:int(len(new_3)//5)]
new_3 = set(new_3)

print(len(new_0), len(new_1), len(new_2))
print('s1')

indices = []

#el2n - [3] - group, [2] - label

d = pd.read_csv('metadata_random_original.csv', index_col=[0])
d = d.reset_index(drop=True)
print(d.shape)
print(d.columns)

for i in tqdm(range(d.shape[0])):
    if (i in new_1) or (i in new_2) or (i in new_3):
        #if (i in new_2) or (i in new_0) or (i in new_3):
        indices.append(i)
print('s1')

print(len(indices))
with open('drop_indices.pkl', 'wb') as f:
    pickle.dump(indices, f)

d = d.drop(index = indices)
print(d.shape)

print('s1')
d.to_csv('metadata_random.csv')
