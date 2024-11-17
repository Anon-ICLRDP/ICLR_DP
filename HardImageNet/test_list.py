import pickle

with open('list.pkl', 'rb') as f:
    x = pickle.load(f)

print(x)
