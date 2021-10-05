trip = {
"123": 0,
"132":1,
"213":2,
"231":3,
"312":4,
"321":5}
doubles = {
"12 ":6,
"21 ":7}
dtot = {
"124":8,
"214":9,
"134":10,
"314":11,
"234":12,
"324":13}
f = {"444":14}
trip_r = {}
dtot_r = {}

for k in trip.keys():
    trip_r[trip[k]] = k
for k in dtot.keys():
    dtot_r[dtot[k]] = k
# Infile
from sys import argv
file1 = argv[1]
# file1 = "150MeV_0k_triples_100k_dtot.csv"
# Outfile
file2 = "sampleout.csv"
from pandas import read_csv
data = read_csv(file1)
d_c  = data.copy()
from numpy import array_split
v   = {"trips": trip,   "dtot": dtot}
v_r = {"trips": trip_r, "dtot": dtot_r}
# Get the data style
I1 = ['e1', 'x1', 'y1', 'z1']
I2 = ['e2', 'x2', 'y2', 'z2']
I3 = ['e3', 'x3', 'y3', 'z3']
I = [I1, I2, I3]
# 213
# e2 <- e1
# e1 <- e2
# e3 <- e3
# e1 -> e2
# e2 -> e1
# e3 -> e3
def dIN(o, n):
    oN = int(o[n-1])
    if oN == 4:
        # 4 will always in the last slot
        # print("--- o == {}".format(o))
        return dI4(I[int(o[0])-1], I[int(o[1])-1])
    else:
        return I[oN-1]

def dI4(D1, D2):
    for D in I:
        if not lEQ(D1, D) and not lEQ(D2, D):
            # print(D1)
            # print(D2)
            # print(D)
            return D

def lEQ(l1, l2):
    if len(l1) != len(l2):
        return False
    for i in range(len(l1)):
        if l1[i] != l2[i]:
            return False
    return True
for t in v.keys():
    # Get all data of that style
    l_class = sorted(v_r[t].keys())[0] # should be 0 or 8
    i = data.loc[data['class'] == l_class, :].index.to_numpy()
    # Split the indices into a section for each order
    k = v[t].keys()
    a = array_split(i, len(k))
    # Now rearrange section into the intended order
    for ind, o in enumerate(k):
        # Sets Interaction ind2 to be whatever dIN determines should be there
        for ind2, oN in enumerate([o[0], [o[1]], o[2]]):
            print("{} :: {} <- {}".format(o, dIN(o, ind2+1), I[ind2]))
            # d_c.loc[a[ind], I[ind2]] = data.loc[a[ind], dIN(o, ind2+1)]
            d_c.loc[a[ind], dIN(o, ind2+1)] = \
                    data.loc[a[ind], I[ind2]].to_numpy()
        d_c.loc[a[ind], 'class'] = v[t][o]
d_c = d_c.sample(frac=1)
d_c.to_csv(file2, index=False)
