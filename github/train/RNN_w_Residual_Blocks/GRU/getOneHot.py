def getOneHot(data, restricted=None, normed=False, triples_only=False, 
        normalization=None, **args):
    from pandas import read_csv
    from numpy import zeros, arange
    from numpy import unique
    if isinstance(data, str):
        print("Data ", data)
        d = read_csv(data)
    else:
        d = data.copy()
    print("Data shape", d.shape)
    print("Columns", d.columns)
    classes = d[["class"]].copy()
    if "euc1" not in d.columns:
        print("Euclidean distance is missing!")
        # Compute Euclidean Distance
        import compton_math
        d = compton_math.euc(d)
        print("Euclidean distance is computed!")
    # Loads in and uses supplied normalizer.... passes DataFrame
    if normalization is not None and isinstance(normalization, str):
        normalization = normalization.split(".")
        baseImport = __import__(normalization[0], globals(), locals(), [normalization[1]], 0)
        normalization = getattr(baseImport, normalization[1])
        # Allows you to access the parameter file
        print("Normalization activated: {}".format(normalization))
        d = normalization(d, **args)
    # Cuts down the classes to the required ones
    if restricted is not None:
        d = d[restricted]

    num_class = len(unique(classes.to_numpy()))
    # Implies that the data is generated with doubles in mind but 
    # contains only # triples
    a = unique(classes.to_numpy(), return_counts=True)
    # print(a)
    # if max(unique(classes.to_numpy())) > 13:
    if num_class < 15:
        classes[classes["class"] >= 8] -= 2 
    classes = classes.to_numpy(dtype=int)
    classes = classes.flatten()
    # print(d.iloc[0,:-1])
    if "class" not in d.columns:
        x = d.iloc[:,:].to_numpy()
    else:
        x = d.iloc[:,:-1].to_numpy()
    # print("x.shape", x.shape)
    mclass = classes.max() + 1 
    rows = x.shape[0]
    y = zeros((rows, mclass))
    # print((rows, mclass))
    l = arange(rows)
    # print(l.shape)
    # print(classes.shape)
    y[arange(rows), classes] = 1
    x = x.reshape(rows, 3, -1)

    print("x.shape", x.shape)
    print("y.shape", y.shape)
    return x, y
