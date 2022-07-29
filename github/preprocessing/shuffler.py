trip = {
"123": 0,
"132":1,
"213":2,
"231":3,
"312":4,
"321":5}
doubles = {
"12":6,
"21":7}
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

from pandas import read_csv
from numpy import array_split, unique
v   = {"trips": trip,   "dtot": dtot}
v_r = {"trips": trip_r, "dtot": dtot_r}

def dIN(o, n, I):
    oN = int(o[n-1])
    if oN == 4:
        # 4 will always in the last slot
        # print("--- o == {}".format(o))
        return dI4(I[int(o[0])-1], I[int(o[1])-1], I)
    else:
        return I[oN-1]

def dI4(D1, D2, I):
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

def passthroughWrapper(kwargs):
    return process_file(**kwargs)

def generateParallelArgs(infiles, kwargs):
    for infile in infiles:
        new_args = kwargs.copy()
        new_args['infile'] = infile
        yield new_args
        

def process_file(infile=None, interaction=None, # outfile=None, 
        not_the_rows=False, not_the_cols=False, prefix=None, verbose=False):
    import pathlib
    writtenFiles = []
    outfile = "{}/{}_shuffled.csv".format(
            prefix,
            pathlib.Path(infile).stem
        )

    # Check interaction contents
    if interaction is None:
        raise("Interactions cannot be empty!")

    # Create interaction groups
    # if data_type == "triples":
    # I1 = ['e1', 'x1', 'y1', 'z1']
    # I2 = ['e2', 'x2', 'y2', 'z2']
    # I3 = ['e3', 'x3', 'y3', 'z3']
    I1 = []
    I2 = []
    I3 = []
    for i, l in enumerate([I1, I2, I3]):
        for col in interaction:
            l.append(col+str(i+1))
    I = [I1, I2, I3]
    # Load in data
    data = read_csv(infile)
    # Shuffle data
    d_c = shuffle_triples(data, I, not_the_rows, not_the_cols, verbose)

    # elif data_type == "doubles":
    # I1 = ['e1', 'x1', 'y1', 'z1']
    # I2 = ['e2', 'x2', 'y2', 'z2']
    I1 = []
    I2 = []
    for i, l in enumerate([I1, I2]):
        for col in interaction:
            l.append(col+str(i+1))
    I = [I1, I2]
    # Load in data
    # Shuffle data
    # data = read_csv(infile)
    # d_c = shuffle_doubles(data, I, not_the_rows, not_the_cols, verbose)
    d_c = shuffle_doubles(d_c, I, not_the_rows, not_the_cols, verbose)
    # else:
        # raise("data_type must be triples or doubles!")
    if verbose:
        print(unique(d_c['class'], return_counts=True))
    
    print(outfile)
    d_c.to_csv(outfile, index=False)
    writtenFiles.append(outfile)
    return writtenFiles

def run(#data_type=None, 
        infiles=None, interaction=None, # outfile=None, 
        not_the_rows=False, not_the_cols=False, prefix=None, verbose=False):

    import pathlib
    if prefix is None:
        prefix = "./"
    writtenFiles = []
    import multiprocessing
    import psutil
    CORES = psutil.cpu_count()
    # CORES = 1
    print("Using", CORES, "cores")
    kwargs = {
            'prefix': prefix,
            'interaction': interaction,
            'not_the_rows': not_the_rows,
            'not_the_cols': not_the_cols,
            'verbose': verbose
        }

    with multiprocessing.Pool(CORES) as p:
        listsolists = p.map(passthroughWrapper, generateParallelArgs(infiles, kwargs))
    writtenFiles = []
    for l in listsolists:
        writtenFiles.extend(l)
    # for infile in infiles:
        # outfile = "{}/{}_shuffled.csv".format(
                # prefix,
                # pathlib.Path(infile).stem
            # )

        # # Check interaction contents
        # if interaction is None:
            # raise("Interactions cannot be empty!")

        # # Create interaction groups
        # # if data_type == "triples":
        # # I1 = ['e1', 'x1', 'y1', 'z1']
        # # I2 = ['e2', 'x2', 'y2', 'z2']
        # # I3 = ['e3', 'x3', 'y3', 'z3']
        # I1 = []
        # I2 = []
        # I3 = []
        # for i, l in enumerate([I1, I2, I3]):
            # for col in interaction:
                # l.append(col+str(i+1))
        # I = [I1, I2, I3]
        # # Load in data
        # data = read_csv(infile)
        # # Shuffle data
        # d_c = shuffle_triples(data, I, not_the_rows, not_the_cols, verbose)

        # # elif data_type == "doubles":
        # # I1 = ['e1', 'x1', 'y1', 'z1']
        # # I2 = ['e2', 'x2', 'y2', 'z2']
        # I1 = []
        # I2 = []
        # for i, l in enumerate([I1, I2]):
            # for col in interaction:
                # l.append(col+str(i+1))
        # I = [I1, I2]
        # # Load in data
        # # Shuffle data
        # # data = read_csv(infile)
        # # d_c = shuffle_doubles(data, I, not_the_rows, not_the_cols, verbose)
        # d_c = shuffle_doubles(d_c, I, not_the_rows, not_the_cols, verbose)
        # else:
            # raise("data_type must be triples or doubles!")
        # if verbose:
            # print(unique(d_c['class'], return_counts=True))
        
        # print(outfile)
        # d_c.to_csv(outfile, index=False)
        # writtenFiles.append(outfile)
    return writtenFiles

def shuffle_doubles(d, I, not_the_rows=False, not_the_cols=False, verbose=False):
    d_true1 = d[(d['class'] == doubles['12'])].index 
    d_true2 = d[(d['class'] == doubles['21'])].index
    # Swap half 
    if not not_the_cols:
        for i, d_true in enumerate([d_true1, d_true2]):
            ind1, ind2 = array_split(d_true.to_numpy(), 2)
            d_copy = d.copy()

            # Swap the 2nd set
            I1, I2 = I
            d.loc[ind2, I1] = d_copy.loc[ind2, I2].to_numpy()
            d.loc[ind2, I2] = d_copy.loc[ind2, I1].to_numpy()
            if i == 0:   # 12 -> 21
                d.loc[ind2, 'class'] = doubles['21']
            elif i == 1: # 21 -> 12
                d.loc[ind2, 'class'] = doubles['12']

    # print("Unique entries after:")
    # print(unique(d['class'].to_numpy()))
    if not not_the_rows:
        d = d.sample(frac=1)
    else:
        if verbose:
            print("Rows not shuffled...")
    return d

def shuffle_triples(data, I, not_the_rows=False, not_the_cols=False, verbose=False):
    # print(I)
    d_c  = data.copy()
    if verbose:
        print("Before shuffling:")
        print(unique(d_c['class'], return_counts=True))
    if not not_the_cols:
        for itype in v.keys():
            if verbose:
                print("Event Type:", itype)
            # Get all data of that style
            l_class = sorted(v_r[itype].keys())[0] # should be 0 or 8
            i = data.loc[data['class'] == l_class, :].index.to_numpy()
            # Split the indices into a section for each order
            tkeys = v[itype].keys()
            a = array_split(i, len(tkeys))
            if verbose:
                print("Index split:", a)
            # Now rearrange section into the intended order
                print("Event Type Keys:", tkeys)
            for ind, o in enumerate(tkeys):
                # Sets Interaction ind2 to be whatever dIN determines should be there
                for ind2, oN in enumerate([o[0], [o[1]], o[2]]):
                    newInter = dIN(o, ind2+1, I)
                    if verbose:
                        print("{} :: {} <- {}".format(o, newInter, I[ind2]))
                    # d_c.loc[a[ind], I[ind2]] = data.loc[a[ind], dIN(o, ind2+1)]
                    d_c.loc[a[ind], newInter] = \
                            data.loc[a[ind], I[ind2]].to_numpy()
                d_c.loc[a[ind], 'class'] = v[itype][o]

    if verbose:
        print("After shuffling:")
        print(unique(d_c['class'], return_counts=True))

    if not not_the_rows:
        d_c = d_c.sample(frac=1)
    else:
        if verbose:
            print("Rows not shuffled...")
    return d_c

def getShufflerParser():
    import argparse
    parser = argparse.ArgumentParser(description="Shuffles interaction data")
    # parser.add_argument("-t", "--data-type", required=True, default=None,
            # choices=["triples", "doubles"],
            # help="""Tells the shuffler how to handle incoming data.
            # Should it shuffle them as doubles or as triples.""")
    parser.add_argument("-i", "--infiles", required=True, default=None,
            nargs="+",
            help="""The filename of the input file to be shuffled.""")
    parser.add_argument("--interaction", required=False, 
            default=['e', 'x', 'y', 'z'],
            type=str,
            nargs="+",
            help="""BY default each interaction is assumed to contain
            eN, xN, yN, zN, and when the interactions are shuffled
            these columns are rearranged as a group.
            If, for some reason, your interactions consist of different
            column names in sets of 2 (for doubles) or 3 (for triples)
            then this argument allows for semi-custom interaction contents."""
            )
    # parser.add_argument("-o","--outfile", required=False,
            # default="sampleout.csv",
            # help="""The outfile filename.
            # If not provided the output file will be called 'sampleout.csv'.
            # """)
    parser.add_argument("-p", "--prefix", required=False,
            default="./",
            help="""
            PREFIX
            """)
    parser.add_argument("-v", "--verbose", required=False,
            default=False,
            action="store_true",
            help="""See the debug noise!""")
    parser.add_argument("--not-the-rows", required=False, default=False,
            action="store_true",
            help="""If only the interactions should be shuffled and not the rows
            please use this argument.""")
    parser.add_argument("--not-the-cols", required=False, default=False,
            action="store_true",
            help="""If only the rows should be shuffled and not the interactions
            please use this argument.""")
    return parser

def main(args=None):
    parser = getShufflerParser()
    args = parser.parse_args(args)
    args = vars(args)
    writtenFiles = run(**args)
    # print(writtenFiles)
    return writtenFiles


if __name__ == "__main__":
    main()
