import pickle
import os

with open('0el2n.pkl', 'rb') as f:
    x = pickle.load(f)

print(len(x))
