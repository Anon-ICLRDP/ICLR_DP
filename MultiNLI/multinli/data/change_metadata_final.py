import pickle
import pandas as pd
import os

from tqdm import tqdm

with open('rank_20_other1.pkl', 'rb') as f:
    all_0, all_1, all_2 = pickle.load(f)
with open('../../majority_indices_final.pkl', 'rb') as f:
    rank = pickle.load(f)

all_corr_indices = []
all_latest_indices = []

for i in rank:
    all_corr_indices.append(i)
print(all_corr_indices[:10])
print(all_corr_indices)
all_corr_indices = [int(i) for i in all_corr_indices]
all_corr_indices = set(all_corr_indices)
#print(all_corr_indices)
#print(max(all_corr_indices))
#print(len(all_corr_indices))
#exit()

d = pd.read_csv('metadata_random_original.csv', index_col=[0])
d = d.reset_index(drop=True)
print(d.shape)
print(d.columns)
#print(d.head(-10))
#exit()

'''indices = []
curr = range(d.shape[0])
relevant = []
for i in tqdm(range(d.shape[0])):
    relevant.append(i)
    if i in all_corr_indices:
        indices.append(i)

print(len(set(relevant).intersection(curr)))
exit()'''

#print('this', len(indices))
#exit()
with open('drop_indices.pkl', 'wb') as f:
    pickle.dump(all_corr_indices, f)
print(d.shape)
d = d.drop(index = all_corr_indices)
print(d.shape)

d.to_csv('metadata_random.csv')
print('worked')
