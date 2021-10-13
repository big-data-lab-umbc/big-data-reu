from sys import argv
from pandas import DataFrame, read_csv
from compton_math import euc

def getMNormalizer(maxabs_normalizer=None):
    from sklearn.preprocessing import MaxAbsScaler
    import pickle
    if maxabs_normalizer is None:
        return MaxAbsScaler(), False
    else:
        with open(maxabs_normalizer, "rb") as infile:
            maxabs_normalizer = pickle.load(infile)
        return maxabs_normalizer, True


def getPNormalizer(power_normalizer=None):
    from sklearn.preprocessing import PowerTransformer
    import pickle
    if power_normalizer is None:
        return PowerTransformer(), False
    else:
        with open(power_normalizer, "rb") as infile:
            power_normalizer = pickle.load(infile)
        return power_normalizer, True

def saveNormalizer(n, root, ext):
    import pickle
    print(root+ext)
    with open(root+ext, "wb") as outfile:
        pickle.dump(n, outfile)

def main(infile=None, 
        maxabs_normalizer=None,
        power_normalizer=None, 
        data_type=None):
    from pandas import DataFrame, read_csv
    import pickle
    ext  = ".csv"
    space = [
        'euc1', 'x1', 'y1', 'z1',
        'euc2', 'x2', 'y2', 'z2',
        ]
    energies = ['e1', 'e2']
    if data_type == "doubles":
        interactions = 2
    elif data_type == "triples":
        interactions = 3
        space.extend([
            'euc3', 'x3', 'y3', 'z3',
            ])
        energies.extend(['e3'])

    if len(energies) % interactions != 0:
        print("Energies does not wrap correctly!")
        print("len(energies) == {}\ninteractions == {}".format(
            len(energies), interactions))
        return
    if len(space) % interactions != 0:
        print("Energies does not wrap correctly!")
        print("len(space) == {}\ninteractions == {}".format(
            len(space), interactions))
        return


    original = read_csv(infile+ext)
    eucframe = euc(original)

    # Energy Transformation
    power_normalizer, ploaded = getPNormalizer(power_normalizer)
    t = eucframe[energies].to_numpy()
    t = t.reshape(-1, int(len(energies)/interactions))
    if not ploaded:
        # fit transform
        t = power_normalizer.fit_transform(t)
        saveNormalizer(power_normalizer, infile, "_power.pickle")
    else:
        # transform
        t = power_normalizer.transform(t)
    t = t.reshape(-1, len(energies))
    eucframe[energies] = t

    maxabs_normalizer, mloaded = getMNormalizer(maxabs_normalizer)
    t = eucframe[space].to_numpy()
    t = t.reshape(-1, int(len(space)/interactions))
    if not mloaded:
        t = maxabs_normalizer.fit_transform(t)
        saveNormalizer(maxabs_normalizer, infile, "_maxab.pickle")
    else:
        t = maxabs_normalizer.transform(t)
    t = t.reshape(-1, len(space))
    eucframe[space] = t
    if mloaded or ploaded:
        outfile = infile+"_same_normed"+ext
    else:
        outfile = infile+"_normed"+ext
    eucframe.to_csv(outfile, index=False)
    print(outfile)
    # print(eucframe)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="""Normalizes data
    given a normalizer for NN. 
    If no normalizer is provided then it creates one.""")
    parser.add_argument("-i", "--infile", 
            required=True, default=None,
            help="""The infile to be processed. 
            The infile is assumed to be a CSV so please do not use
            the .csv extension in the filename.""")
    parser.add_argument("-pn", "--power-normalizer", 
            required=False, default=None,
            help="""The location of the power transformer to be used.
            If none is provided one is generated and saved.""")
    parser.add_argument("-mn", "--maxabs-normalizer", 
            required=False, default=None,
            help="""The location of the max abs scaler to be used.
            If none is provided one is generated and saved.""")
    parser.add_argument("-t", "--data-type", 
            required=True, default=None,
            help="""The type of scattering data. Triples or doubles.
            """)
    args = parser.parse_args()
    args = vars(args)
    main(**args)

