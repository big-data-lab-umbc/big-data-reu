from pandas import read_csv
from numpy import unique

c = {"12":6, "21": 7}
c_rev = {6: "12", "21": 7}

from sys import argv
file1 = argv[1]
ext = ".csv"

d = read_csv(file1+ext)
# print("Unique entries before:")
# print(unique(d['class'].to_numpy()))

# Select all non-false data
d_true = d[d['class'] != 14].index
from numpy import array_split, arange
ind1, ind2 = array_split(d_true.to_numpy(), 2)

d_copy = d.copy()

# Swap the 2nd set
I1 = ['e1', 'x1', 'y1', 'z1']
I2 = ['e2', 'x2', 'y2', 'z2']
d.loc[ind2, I1] = d_copy.loc[ind2, I2].to_numpy()
d.loc[ind2, I2] = d_copy.loc[ind2, I1].to_numpy()
d.loc[ind2, 'class'] = 7

# print("Unique entries after:")
# print(unique(d['class'].to_numpy()))
d.sample(frac=1.0).to_csv(file1+"_shuffled"+ext, index=False)

