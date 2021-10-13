from numpy import isnan
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

labels = [
# Triples
'123',
'132',
'213',
'231',
'312',
'321',
# Doubles
'12',
'21',
# DtoT 412 is specially for DtoT events
'124',
'214',
'134',
'314',
'234',
'324',
# False
'444']
index1 = {}
index2 = {}
for i in range(len(labels)):
    index1[labels[i]] = i
    index2[i] = labels[i]

def loadNormalizers(pow_norm=None, max_norm=None, 
        pow_norm_doubles=None, max_norm_doubles=None):
    import pickle
    # Load triple normalizer
    if pow_norm is not None:
        print(pow_norm)
        with open(pow_norm, "rb") as inNorm:
            powNorm = pickle.load(inNorm)
    if max_norm is not None:
        print(max_norm)
        with open(max_norm, "rb") as inNorm:
            maxNorm = pickle.load(inNorm)

    # Load double normalizer
    # If double normalizer not found then use triple normalizer
    if pow_norm_doubles is not None:
        with open(pow_norm_doubles, "rb") as inNorm:
            powNormDoubles = pickle.load(inNorm)
    else:
        powNormDoubles = powNorm

    if max_norm_doubles is not None:
        with open(max_norm_doubles, "rb") as inNorm:
            maxNormDoubles = pickle.load(inNorm)
    else:
        maxNormDoubles = maxNorm

    if pow_norm is None:
        powNorm = powNormDoubles
    if max_norm is None:
        maxNorm = maxNormDoubles
    return powNorm, maxNorm, powNormDoubles, maxNormDoubles


def getTFModel(model):
    # print(model)
    from tensorflow.keras.models import load_model
    from tensorflow import distribute as D
    from tensorflow import config
    devices = [device for device in config.list_physical_devices() if "GPU" == device.device_type]
    devices = ["/gpu:{}".format(i) for i, device in enumerate(devices)]
    # model = '../unencoded/testNetwork_maxabs_euc/model-final/'
    # model = load_model(model)
    # First resolve directory... 
    if model is not None:
        from pathlib import Path
        p = Path(model)
        if not p.is_dir():
            print(p, p.is_dir())
            return None
    else:
        return None
    if len(devices) > 1:
        strat = D.MirroredStrategy(devices=devices,
                cross_device_ops=D.HierarchicalCopyAllReduce())
        with strat.scope():
            model     = load_model(model)
    else:
        model     = load_model(model)
    return model


def loadNetwork(network):
    return getTFModel(network)

def normWithWrapping(inframe, powNorm, maxNorm, isDoubles=False):
    import pickle
    from compton_math import euc
    # First we compute euclidean distance
    eucFrame = euc(inframe)
    normFrame = eucFrame.copy()
    # First we normalize by not energy
    if isDoubles:
        space = [
                'euc1', 'x1', 'y1', 'z1',
                'euc2', 'x2', 'y2', 'z2',
                ]
        fullspace = [
                'euc1','e1', 'x1', 'y1', 'z1',
                'euc2','e2', 'x2', 'y2', 'z2',
                ]
        energies = ['e1', 'e2']
        interactions = 2
    else:
        space = [
                'euc1', 'x1', 'y1', 'z1',
                'euc2', 'x2', 'y2', 'z2',
                'euc3', 'x3', 'y3', 'z3',
                ]
        fullspace = [
                'euc1','e1', 'x1', 'y1', 'z1',
                'euc2','e2', 'x2', 'y2', 'z2',
                'euc3','e3', 'x3', 'y3', 'z3',
                ]
        energies = ['e1', 'e2', 'e3']
        interactions = 3
    print(eucFrame.columns)
    print(eucFrame.shape)
    t = eucFrame[space]
    print(t.columns, t.shape)
    t = t.to_numpy().reshape(-1, int(len(space)/interactions))
    print(t.shape)
    print("Max norm:", maxNorm)
    t = maxNorm.transform(t)
    t = t.reshape(-1, len(space))
    normFrame[space] = t
    # Then we normalize the energies
    t = eucFrame[energies].to_numpy().reshape(-1, 1)
    print("Power norm:", powNorm)
    t = powNorm.transform(t)
    t = t.reshape(-1, interactions)
    normFrame[energies] = t
    return normFrame[fullspace]

def correctData(inframe, classes):
    from pandas import DataFrame
    from numpy import empty, where, NaN, arange
    fixed = DataFrame(empty(inframe.shape), 
            columns=inframe.columns.to_numpy().tolist())
    indicies = arange(inframe.shape[0])#.reshape(-1, 1)
    fixed.loc[:,:] = NaN
    I1 = ['e1', 'x1', 'y1', 'z1'] 
    I2 = ['e2', 'x2', 'y2', 'z2'] 
    I3 = ['e3', 'x3', 'y3', 'z3']
    I = {"1": I1, "2": I2, "3": I3}
    for cl in labels[:-1]:
        c = classes == index1[cl]
        c = indicies[c]
        # print(c)
        # print(I1)
        # print(I[cl[0]])
        # 1
        fixed.loc[c, I1]  = inframe.loc[c, I[cl[0]]].to_numpy()
        # 2
        fixed.loc[c, I2]  = inframe.loc[c, I[cl[1]]].to_numpy()
        # 3
        if len(cl) == 2 or cl[2] == "4":
            # print(cl, "is NN4 or double")
            fixed.loc[c, I3] = NaN
        else:
            fixed.loc[c, I3]  = inframe.loc[c, I[cl[2]]].to_numpy()
        # if c.shape[0] > 0:
            # print(cl, c.shape)
            # print(c)
    c = indicies[classes != 14]
    # print(c)

    fixed = fixed.iloc[c]
    # print(inframe)
    # print(fixed)
    return fixed

def main(infile=None, outfile=None, save_unclean=False, verbose=False,
        network=None, network_doubles=None,
        pow_norm=None, max_norm=None, pow_norm_doubles=None, max_norm_doubles=None):

    # Load normalizers
    powNorm, maxNorm, powNormDoubles, maxNormDoubles = \
            loadNormalizers(pow_norm, max_norm, pow_norm_doubles, max_norm_doubles)

    # Load data
    from pandas import read_csv
    fullspace = [
         'e1', 'x1', 'y1', 'z1', 
         'e2', 'x2', 'y2', 'z2', 
         'e3', 'x3', 'y3', 'z3']
    # Read in first line of file
    with open(infile, "r") as tmp:
        h = tmp.readline()
    h = h.strip().split(",")
    if "class" in h:
        h.remove("class")
        containsClass = True
    else:
        containsClass = False

    if len(h) == len(fullspace) and h == fullspace:
        # Header exists
        print("Header detected!")
        inframe = read_csv(infile, float_precision="high")
    else:
        # No header
        print(h, fullspace)
        inframe = read_csv(infile, float_precision="high", header=None,
                names=fullspace)

    # Remove all singles first
    from numpy import isnan
    from numpy import zeros
    inframe = inframe.iloc[~(isnan(inframe['e2']).to_numpy()), :].copy()
    inframe.reset_index()
    # Setup data structures
    # normFrame = inframe.copy()
    classes = zeros((inframe.shape[0],))
    # bs = 8*65535
    bs = 4*8192
    # Select doubles
    ind_doubles = isnan(inframe['e3']).to_numpy()
    ind_triples = ~ind_doubles
    # Normalize Data
    from numpy import unique
    if network_doubles is not None:
        # Make this an input ...
        # train_space = [
             # 'euc1', 'e1', 'x1', 'y1', 'z1', 
             # 'euc2', 'e2', 'x2', 'y2', 'z2', 
             # ]
        # Load doubles network
        print(network_doubles)
        isDoubles = True
        train_data = \
                normWithWrapping(inframe.iloc[ind_doubles, :], 
                        powNormDoubles, maxNormDoubles, isDoubles)
        modelDoubles = loadNetwork(network_doubles)
        # Make prediction on data
        print(train_data.columns)
        classes[ind_doubles] = \
                modelDoubles.predict(train_data.to_numpy(), batch_size=bs, verbose=0) \
                                .argmax(axis=1)
        # Adjust for triples
        # print(classes[ind_doubles].shape)
        # print(unique(classes[ind_doubles]))
        classes[ind_doubles & (classes == 0)] = index1["12"]
        classes[ind_doubles & (classes == 1)] = index1["21"]
        classes[ind_doubles & (classes == 2)] = index1["444"]
        # print(unique(classes[ind_doubles]))
    else:
        inframe = inframe.iloc[ind_triples, :].copy()
        classes = classes[ind_triples]
    # Select triples
    if network is not None:
        # Load triples network
        # Normalize Data
        print(network)
        isDoubles = False
        train_data = \
                normWithWrapping(inframe.iloc[ind_triples, :], 
                        powNorm, maxNorm, isDoubles)
        model = loadNetwork(network)
        # Make prediction on data
        classes[ind_triples] = model.predict(train_data.to_numpy(),
                batch_size=bs, verbose=2) \
                .argmax(axis=1)
        # Adjust triples to have space for doubles
        classes[ind_triples & (classes >= index1["12"] )] += 2
    else:
        inframe = inframe.iloc[ind_doubles, :].copy()
        classes = classes[ind_doubles]
    # Clean data
    correct = correctData(inframe[fullspace], classes)
    
    from pathlib import Path
    p = Path(infile)
    from numpy import savetxt
    savetxt("{}_classes.csv".format(p.stem), classes, delimiter=',',
            header='class')
    c, counts = unique(classes, return_counts=True)
    with open("{}_class_counts.txt".format(p.stem), 'w') as writefile:
        space = max(list(map(lambda x: len(str(x)), counts)))
        fmt = "{}: {:" + str(space) + "d}"
        final_counts = "\n".join(list(map(lambda m: fmt.format(index2[c[m]], counts[m]),
            range(len(c)))))
        # writefile.write(str(unique(classes, return_counts=True)))
        writefile.write(final_counts)
        writefile.write("\n") 
    # In the class that the DF has class answer dump those too....
    if containsClass:
        c, counts = unique(inframe['class'], return_counts=True)
        space = max(list(map(lambda x: len(str(x)), counts)))
        fmt = "{}: {:" + str(space) + "d}"
        final_counts = "\n".join(list(map(lambda m: fmt.format(index2[c[m]], counts[m]),
            range(len(c)))))
        # writefile.write(str(unique(classes, return_counts=True)))
        with open("{}_true_class_counts.txt".format(p.stem), 'w') as writefile:
            writefile.write(final_counts)
            writefile.write("\n") 
    # Save data
    if outfile is None:
        # Leaf
        ofile = "{}_cleaned.csv".format(p.stem)
        if save_unclean:
            ofile2 = "{}_uncleaned.csv".format(p.stem)
    else:
        ofile = outfile[i]
    correct.to_csv(ofile, index=False, float_format="%24.16f")
    if save_unclean:
        inframe[fullspace].to_csv(ofile2, index=False, float_format="%24.16f")
        if verbose:
            print(ofile2)
    if verbose:
        print(ofile)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generates a confusion matrix given an appropriate CSV and a model.")
    parser.add_argument("-i", "--infile", type=str, required=True, 
            help="Input files must be a compton camera CSV"
            )
    parser.add_argument("-o", "--outfile", type=str, required=False,
            default=None,
            help="""Default output filenames are INFILE_cleaned.csv .
            If multiple infiles are provided then the outfile names should be
            1:1 .
            """
            )
    parser.add_argument("-n", "--network", type=str, required=False,
            help="""Requires a tensorflow neural network directory which \
            contains a .pb file. 
            The network should specialize in triple classification.""")
    parser.add_argument("-nd", "--network-doubles", type=str, required=False,
            default=None,
            help="""Requires a tensorflow neural network directory which
            contains a .pb file.
            The network should specialize in double classification.""")
    parser.add_argument("-mn", "--max-norm", type=str, required=False,
            default=None,
            help="The pickled normalizer that will be used for triples. Probably a MaxAbsScaler()")
    parser.add_argument("-pn", "--pow-norm", type=str, required=False,
            default=None,
            help="""The pickled normalizer that will be used for triples.
            Probably a PowerTransformer()""")
    parser.add_argument("-mnd", "--max-norm-doubles", type=str, required=False,
            default=None,
            help="""The pickled normalizer that will be used for doubles.
            Probably a MaxAbsScaler().
            If doubles are detected and a double network is supplied but
            a double specific normalizer is not supplied then the 
            triple normalizer will be used instead.""")
    parser.add_argument("-pnd", "--pow-norm-doubles", type=str, required=False,
            default=None,
            help="""The pickled normalizer that will be used for doubles.
            Probably a PowerTransformer().
            If doubles are detected and a double network is supplied but
            a double specific normalizer is not supplied then the 
            triple normalizer will be used instead.""")
    parser.add_argument("--save-unclean", required=False,
            default=False, action="store_true",
            help="""If you failed to strip the file this saves a 'stripped'
            copy for you. 
            Only usable if you did not use -o .""")
    parser.add_argument("-v", "--verbose", required=False, default=False,
            action="store_true",
            help="""Determines the verbosity. 
            When verbosity is turned on it will print the output filenames.""")
    args = parser.parse_args()
    args = vars(args)
    main(**args)
