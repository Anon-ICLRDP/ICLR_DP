import pandas as pd
import numpy as np
import os
import pickle


with open('../../50_percent_n.pkl', 'rb') as f:
    x1 = pickle.load(f)

with open('../../75_percent_n.pkl', 'rb') as f:
    x2 = pickle.load(f)

with open('../../90_percent_n.pkl', 'rb') as f:
    x3 = pickle.load(f)

with open('../../95_percent_n.pkl', 'rb') as f:
    x4 = pickle.load(f)

with open('../../97_percent_n.pkl', 'rb') as f:
    x5 = pickle.load(f)

#with open('../../99_percent_n.pkl', 'rb') as f:
#    x6 = pickle.load(f)
with open('../../25_percent_n.pkl', 'rb') as f:
    x6 = pickle.load(f)

with open('../../40_percent_n.pkl', 'rb') as f:
    x7 = pickle.load(f)

with open('../../10_percent_n.pkl', 'rb') as f:
    x8 = pickle.load(f)


m = pd.read_csv('metadata_main.csv')
m_other = pd.read_csv('list_eval_partition_main.csv')
print('og', m.shape)
print('og', m_other.shape)

#m1_new = m.drop(x1, inplace=True)
#m2_new = m.drop(x2, inplace=True)

m1_new = m[~m.index.isin(x1)]
m2_new = m[~m.index.isin(x2)]
m3_new = m[~m.index.isin(x3)]
m4_new = m[~m.index.isin(x4)]
m5_new = m[~m.index.isin(x5)]
m6_new = m[~m.index.isin(x6)]
m7_new = m[~m.index.isin(x7)]
m8_new = m[~m.index.isin(x8)]

print(m1_new.shape)
print(m2_new.shape)
print(m3_new.shape)
print(m4_new.shape)
print(m5_new.shape)
print(m6_new.shape)
print(m7_new.shape)
print(m8_new.shape)
m1_new.to_csv('metadata_50prunen.csv')
m2_new.to_csv('metadata_75prunen.csv')
m3_new.to_csv('metadata_90prunen.csv')
m4_new.to_csv('metadata_95prunen.csv')
m5_new.to_csv('metadata_97prunen.csv')
m6_new.to_csv('metadata_25prunen.csv')
m7_new.to_csv('metadata_40prunen.csv')
m8_new.to_csv('metadata_10prunen.csv')

m1_new = m_other[~m_other.index.isin(x1)]
m2_new = m_other[~m_other.index.isin(x2)]
m3_new = m_other[~m_other.index.isin(x3)]
m4_new = m_other[~m_other.index.isin(x4)]
m5_new = m_other[~m_other.index.isin(x5)]
m6_new = m_other[~m_other.index.isin(x6)]
m7_new = m_other[~m_other.index.isin(x7)]
m8_new = m_other[~m_other.index.isin(x8)]

print(m1_new.shape)
print(m2_new.shape)
print(m3_new.shape)
print(m4_new.shape)
print(m5_new.shape)
print(m6_new.shape)
print(m7_new.shape)
print(m8_new.shape)

m1_new.to_csv('list_eval_partition_50prunen.csv')
m2_new.to_csv('list_eval_partition_75prunen.csv')
m3_new.to_csv('list_eval_partition_90prunen.csv')
m4_new.to_csv('list_eval_partition_95prunen.csv')
m5_new.to_csv('list_eval_partition_97prunen.csv')
m6_new.to_csv('list_eval_partition_25prunen.csv')
m7_new.to_csv('list_eval_partition_40prunen.csv')
m8_new.to_csv('list_eval_partition_10prunen.csv')
