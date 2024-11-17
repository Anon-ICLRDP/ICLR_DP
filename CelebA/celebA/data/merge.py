import torch
import pandas as pd
import numpy as np
import os

df1 = pd.read_csv('metadata.csv')
df2 = pd.read_csv('list_eval_partition.csv')

l1 = list(df2['split'])
df1['split'] = l1

df1.to_csv('mmetadata.csv')
