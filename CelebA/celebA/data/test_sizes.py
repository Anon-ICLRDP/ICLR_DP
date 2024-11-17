import pandas as pd
import numpy as np


d = pd.read_csv('mmetadata.csv')

count = 0

for i in range(d.shape[0]):
    if d.loc[i, 'split'] == 0:
        count += 1

print(count)
