counts = [57498, 11158, 67376, 1521, 66630, 1992]
accs = [0.8496690988540649,0.87077397108078,0.8059810400009155,0.7776525020599365,0.7660167217254639,0.7587109208106995]
s = sum(counts)
t = 0
for i, j in zip(counts, accs):
    t += i*j

print(t/s)