def convertFracToFloat(items):
    newItems = []
    for i, item in enumerate(items):
        if "/" in item:
            args = item.replace(",", "").split("/")
            if len(args) > 0:
                v = float(args[0])
                for arg in args[1:]:
                    v /= float(arg)
                newItems.append(str(v))
                if i != len(items) - 2:
                    newItems[-1] = newItems[-1] + ","
        else:
            newItems.append(item)
    # print(newItems)
    return newItems

def listJSONtoDict(listJSON):
    import json
    
    j = " ".join(convertFracToFloat(listJSON)).replace("'", '"')
    # print(j)
    j = json.loads(
            j
            )
    return j


def getMeVsAndkMUs(filenames):
    import re
    MeVFinder =  re.compile("\d+MeV")
    MeVs = sorted(list(set([z[0][:-3] for z in map(lambda x: MeVFinder.findall(x), filenames)])))
    kMUFinder = re.compile("\d+kMU")
    kMUs = sorted(list(set([z[0][:-3] for z in map(lambda x: kMUFinder.findall(x), filenames)])))
    return MeVs, kMUs

def numToMetricString(string):
    # Assume that the number is in string form
    num = int(string)
    # Determine power
    import math
    maxPower = math.log(num)/math.log(10)
    if maxPower <= 3:
        s = "{:3d}".format(num)
    elif 3 < maxPower and maxPower <  6:
        s = "{:3d}k".format(int(float(num)/10**3))
    elif 6 <= maxPower and maxPower < 9:
        s = "{:3d}M".format(int(float(num)/10**6))
    elif 9 <= maxPower and maxPower < 12:
        s = "{:3d}B".format(int(float(num)/10**9))
    elif 12 <= maxPower and maxPower < 15:
        s = "{:3d}T".format(int(float(num)/10**12))
    return s

def getExtraFilenamePieces(filename):
    import re, pathlib
    strFinder = re.compile("\d+MeV|\d+kMUmin|_|-|singles|doubles|triples|dtot|false|converted")
    if isinstance(filename, pathlib.Path):
        filename = str(filename.stem)
    elif isinstance(filename, str):
        filename = str(pathlib.Path(filename).stem)
    s = list(filter(lambda v: v != '', 
            strFinder.split(filename)))
    return s

def getAllExtraFilenamePieces(filenames):
    allPieces = set()
    for filename in filenames:
        allPieces.add(tuple(getExtraFilenamePieces(filename)))
    return list(allPieces)

def getAutoName(allFiles, allCounts, order, filename_pieces=None):
    # allFiles is the dictionary of filenames created by user input
    # print(allFiles)
    # print(allCounts)
    # print(order)
    filenames = []
    for key in order:
        tList = allFiles[key]
        if tList is not None:
            filenames.extend(tList)
    MeVs, kMUs = getMeVsAndkMUs(filenames)
    if len(MeVs) > 1:
        # Instead choose the total number of events
        MeV = "{}events_{}-multi-MeV".format(
                numToMetricString(sum([allCounts[v] for v in order])),
                len(MeVs))
    else:
        MeV ="{}MeV".format(MeVs[0])
    if len(kMUs) > 1:
        # Instead choose the words "multi-kMU"
        kMU = "{}-multi-kMU".format(len(kMUs))
    else:
        kMU = "{}kMU".format(kMUs[0])
        
    scatter = "_".join(order)

    base = "{}_{}_{}".format(MeV, kMU, scatter).strip()
    if filename_pieces is not None and len(filename_pieces) > 0:
        base += "_" + "_".join(filename_pieces)
        

    return "{}.csv".format(base)

def getAllFrames(infiles, oldFrame=None, verbose=False):
    import pandas
    # Load first frame
    if len(infiles) > 0:
        if oldFrame is not None:
            bigFrame = pandas.concat([
                oldFrame,
                pandas.read_csv(infiles[0])
            ])
        else:
            bigFrame = pandas.read_csv(infiles[0]) 

        for infile in infiles[1:]:
            bigFrame = pandas.concat([
                bigFrame,
                pandas.read_csv(infile)
            ])

        return bigFrame
    else:
        if oldFrame is not None:
            return oldFrame
        else:
            return None

def flattenFilenameDictionary(filenameDictionary, order, return_valid_keys=False):
    allFilenames = []
    allKeys = []
    for key in order:
        tList = filenameDictionary[key]
        if tList is not None:
            allKeys.append(key)
            allFilenames.extend(tList)
    if return_valid_keys:
        return allFilenames,  allKeys
    else:
        return allFilenames

def getMeVkMUFiles(MeV="none", kMU="none", allFiles=None):
    # allFiles is a list
    import re
    pattern1 = re.compile(r"([^0-9]|^|[0-9.+]*[a-zA-Z._])({}MeV)".format(MeV))
    pattern2 = re.compile(r"([^0-9]|^|[0-9.+]*[a-zA-Z._])({}kMU)".format(kMU))
    matchedFiles = []
    passed = False
    if MeV == "none" and kMU == "none":
        return allFiles
    
    for f in allFiles:
        if MeV != "none":
            search1 = pattern1.search(f)
        else:
            search1 = True

        if kMU != "none":
            search2 = pattern2.search(f)
        else:
            search2 = True
        
        if search1 and search2:
            matchedFiles.append(f)

    return matchedFiles

def getMatchingFilenames(filenames, pieces):
    if isinstance(pieces, str) and pieces == "none":
        return filenames
    if isinstance(pieces, tuple) and len(pieces) == 0:
        return filenames

    matchedFiles = []
    for filename in filenames:
        matched = True
        for piece in pieces:
            if piece not in filename:
                matched = False
        if matched:
            matchedFiles.append(filename)

    return matchedFiles

def createSubsets(preferences=None, filenames=None, write_results_to_disk=False):
    # Filenames is a dict or lists
    if preferences is None:
        preferences = ["none"]
    if "none" in preferences and len(preferences) > 1:
        print("Conflicting merge preference provided.")
        print('"none" is an overriding preference.')
        print('Defaulting to "none"')
        preferences = ["none"]
    # choices=["none", "mev", "kmu", "scatter"]
    # Have
    {
        "triples": [
            "filename1",
            "filename2",
            "filename3",
            # ...
            ],
        "dtot": [
            "filename4",
            # ...
            ]
        # ...

    }
    # Want
    {
            '000MeV': {
                "000kMU": {
                        "triples": [
                            "filename1",
                            "filename2",
                            "filename3",
                            # ...
                            ],
                        "dtot": [
                            "filename4",
                            # ...
                            ]
                    }   
            } 
            
    }
    allFilenames, scatters = flattenFilenameDictionary(filenames, filenames.keys(),
            return_valid_keys=True)

    MeVs, kMUs = getMeVsAndkMUs(
            allFilenames
            )
    # print(preferences)
    # print(MeVs)
    # print(kMUs)
    c = {}
    if 'mev' in preferences:
        for MeV in MeVs:
            c[MeV] = {}
    else:
        c["none"] = {}

    for MeV in c.keys():
        if 'kmu' in preferences:
            for kMU in kMUs:
                c[MeV][kMU] = {}
        else:
            c[MeV]["none"] = {}
    
    allFilepieces = getAllExtraFilenamePieces(allFilenames)
    # print(allFilepieces)
    for MeV in c.keys():
        for kMU in c[MeV].keys():
            if 'amf' in preferences:
                for piece in allFilepieces:
                    c[MeV][kMU][piece] = {}
            else:
                c[MeV][kMU][tuple()] = {}
    for MeV in c.keys():
        for kMU in c[MeV].keys():
            for pieces in c[MeV][kMU].keys():
                for scatter in scatters:
                    c[MeV][kMU][pieces][scatter] = \
                        getMatchingFilenames(
                            getMeVkMUFiles(MeV=MeV, kMU=kMU, 
                                    allFiles=filenames[scatter]), pieces)

                # print(allFiles[scatter])
    # write_results_to_disk = True
    if write_results_to_disk:
        with open("resultingSubset.txt", "w") as outfile:
            tabs = "    "
            outfile.write("{\n")
            for MeV in c.keys():
                outfile.write(1*tabs + "{}:\n".format(MeV))
                for kMU in c[MeV].keys():
                    outfile.write(2*tabs + "{}:\n".format(kMU))
                    for pieces in c[MeV][kMU].keys():
                        outfile.write(3*tabs+ "{}:\n".format(pieces))
                        for scatter in c[MeV][kMU][pieces].keys():
                            outfile.write(4*tabs + "{}:\n".format(scatter))
                            for filename in c[MeV][kMU][pieces][scatter]:
                                outfile.write(5*tabs + str(filename))
                                outfile.write("\n")
    return c

def process_files(prefix=None,
        json_ratio=None,
        verbose=False,
        undersample=True,
        random=False,
        outfile=None,
        filename_pieces=(),
        separate_scatters=False,
        **kwargs
        ):
    #######
    import numpy
    import pandas
    bigFrame = None
    order = []
    allCounts = {}
    inds = {}
    lenBefore = 0
    writtenFiles = []
    for key in kwargs:
        # print(key)
        tList = kwargs[key]

        if bigFrame is None and tList is not None:
            bigFrame = getAllFrames(tList, None, verbose)
            if bigFrame is None:
                print(kwargs)
            lenBefore = bigFrame.shape[0]
            inds[key] = numpy.arange(lenBefore)
            allCounts[key] = lenBefore
            order.append(key)
            if verbose:
                print("bigFrame.shape", bigFrame.shape)
        elif bigFrame is not None and tList is not None:
            bigFrame = getAllFrames(tList, bigFrame, verbose)
            allCounts[key] = bigFrame.shape[0] - lenBefore
            inds[key] = numpy.arange(start=lenBefore, stop=bigFrame.shape[0])
            lenBefore = bigFrame.shape[0]
            order.append(key)
            if verbose:
                print("bigFrame.shape", bigFrame.shape)
    if verbose:
        print(inds)
        # Under sample if requested
        print(bigFrame.shape)
        # classes, counts = numpy.unique(bigFrame["class"], return_counts=True)
        print(allCounts)
    if undersample:
        minEvents = None
        for key in allCounts.keys():
            # print(key)
            # print(json_ratio[key])
            # print(allCounts[key])
            num = json_ratio[key]
            if minEvents is None and num != 0:
                minEvents = int(allCounts[key]*(num**-1))
            elif num != 0:
                minEvents = min([minEvents, int(allCounts[key]*(num**-1))])
        if verbose:
            print(minEvents)
        
        # Now select the new items
        for i, scatter in enumerate(order):
            f = inds[scatter]
            # This undersamples every scatter type
            if random:
                f = numpy.random.choice(f, size=int(minEvents*json_ratio[scatter]), replace=False)
            else:
                f = f[:int(minEvents*json_ratio[scatter])]
            inds[scatter] = f
        # print(sum([minEvents*json_ratio[v] for v in order]))
    
    # Generate the array that will be used for selecting the rows
    # When undersampling has been performed arr will be a subset of the
    # collection of all indices
    # When undersampling has NOT been performaed arr will be equal to the
    # collection of all indices
    arr = numpy.concatenate([inds[scatter] for scatter in order])
    if separate_scatters:
        for scatter in order:
            if outfile is None:
                outname = getAutoName({scatter: kwargs[scatter]}, 
                        {scatter: allCounts[scatter]}, [scatter],
                        filename_pieces)
            else:
                outname = outfile
            outname = "{}/{}".format(prefix, outname)
            outFrame = bigFrame.iloc[inds[scatter], :]
            if verbose:
                classes, counts = numpy.unique(outFrame["class"], return_counts=True)
                print(outFrame.shape)
                for i in range(classes.shape[0]):
                    print("{:2d}: {:9d}".format(classes[i], counts[i]))
            print(outname)
            outFrame.to_csv(outname, index=False)
            writtenFiles.append(outname)
            # print(numpy.unique(outFrame["class"], return_counts=True))
            # outFrame.to_csv(outname)
            # outFrame.iloc[inds[scatter], :].to_csv(outname)
    else:
        # Save to disk
        if outfile is None:
            outname = getAutoName(kwargs, allCounts, order,
                    filename_pieces)
        else:
            outname = outfile
        outname = "{}/{}".format(prefix, outname)
        print(outname)
        # outFrame.to_csv(outname)
        # outFrame.iloc[arr, :].to_csv(outname)
        outFrame = bigFrame.iloc[arr, :]
        if verbose:
            print(outFrame.shape)
            classes, counts = numpy.unique(outFrame["class"], return_counts=True)
            for i in range(classes.shape[0]):
                print("{:2d}: {:9d}".format(classes[i], counts[i]))
        outFrame.to_csv(outname, index=False)
        writtenFiles.append(outname)
    return writtenFiles

def run(
        prefix=None,
        json_ratio=None,
        verbose=False,
        undersample=True,
        random=False,
        outfile=None,
        parallel=False,
        merge_preferences=None,
        write_results_to_disk=False,

        # triples=None, dtot=None, false_triples=None,
        # doubles=None, false_doubles=None,
        **kwargs
        ):
    
    import pathlib
    p = pathlib.Path(prefix)
    if not p.is_dir():
        print("Prefix does not exist, I will create it...")
        p.mkdir(exist_ok=True, parents=True)
        print(str(p))

    if json_ratio is not None:
        json_ratio = listJSONtoDict(json_ratio.split())
    elif undersample:
        print("Undersampling desired but no ratio present? Exiting!")
        exit(1)

    subsets = createSubsets(preferences=merge_preferences, filenames=kwargs,
            write_results_to_disk=write_results_to_disk)
            # write_results_to_disk=True)
    # exit()
    # print(subsets)
    if "scatters" in merge_preferences:
        separate_scatters = True
    else:
        separate_scatters = False
    writtenFiles = []
    # for MeV in subsets.keys():
        # for kMU in subsets[MeV].keys():
            # for pieces in subsets[MeV][kMU].keys():
                # readySubset = subsets[MeV][kMU][pieces]
                # totalFiles = sum([len(readySubset[key]) for key in readySubset.keys()])
                # if totalFiles > 0:
                    # someFiles = \
                        # process_files(prefix=prefix, 
                            # json_ratio=json_ratio, 
                            # verbose=verbose, 
                            # undersample=undersample, 
                            # random=random, 
                            # outfile=outfile,
                            # separate_scatters=separate_scatters, 
                            # filename_pieces=pieces,
                            # **readySubset)
                    # writtenFiles.extend(someFiles)
                # else:
                    # if verbose:
                        # print(MeV, kMU, readySubset)
    if parallel:
        import multiprocessing
        import psutil
        CORES = psutil.cpu_count()
        # CORES = 1
        print("Using", CORES, "cores")
        kwargs = {
                'prefix':            prefix, 
                'json_ratio':        json_ratio, 
                'verbose':           verbose, 
                'undersample':       undersample, 
                'random':            random, 
                'outfile':           outfile,
                'separate_scatters': separate_scatters,  
            }

        with multiprocessing.Pool(CORES) as p:
            listsolists = p.map(passthroughWrapper, generateParallelSubsets(subsets, kwargs))
        writtenFiles = []
        for l in listsolists:
            writtenFiles.extend(l)
    else:
        for readySubset, pieces in getReadySubset(subsets):
            someFiles = \
                process_files(prefix=prefix, 
                    json_ratio=json_ratio, 
                    verbose=verbose, 
                    undersample=undersample, 
                    random=random, 
                    outfile=outfile,
                    separate_scatters=separate_scatters, 
                    filename_pieces=pieces,
                    **readySubset)
            writtenFiles.extend(someFiles)
        # if verbose:
            # print(CORES)
    # return writtenFiles

    return writtenFiles


def passthroughWrapper(kwargs):
    return process_files(**kwargs)

def generateParallelSubsets(subsets, kwargs):
    for subset, pieces in getReadySubset(subsets):
        newDict = kwargs.copy()
        for key in subset.keys():
            newDict[key] = subset[key]
        newDict['filename_pieces'] = pieces
        yield newDict

def getReadySubset(subsets):
    for MeV in subsets.keys():
        for kMU in subsets[MeV].keys():
            for pieces in subsets[MeV][kMU].keys():
                # yield pieces
                readySubset = subsets[MeV][kMU][pieces]
                totalFiles = sum([len(readySubset[key]) for key in readySubset.keys()])
                if totalFiles > 0:
                    yield readySubset, pieces

def getMergerParser():
    import argparse
    parser = argparse.ArgumentParser(description="Shuffles interaction data")
    parser.add_argument("--triples", required=False,
            nargs="+")
    parser.add_argument("--false-triples", required=False,
            nargs="+")
    parser.add_argument("--dtot", required=False,
            nargs="+")
    parser.add_argument("--doubles", required=False,
            nargs="+")
    parser.add_argument("--false-doubles", required=False,
            nargs="+")
    parser.add_argument("-p", "--prefix", required=False,
            default="./",
            help="""
            PREFIX
            """)
    parser.add_argument("-s", "--as-is",
            dest="undersample",
            action="store_false",
            default=True
            )
    parser.add_argument("-j", "--json-ratio", required=False,
            type=str,
            default=None,
            # nargs=1
            )
    parser.add_argument("-v", "--verbose", required=False,
            default=False,
            action="store_true",
            help="""See the debug noise!""")
    parser.add_argument("--merge-preferences", required=False,
            choices=["none", "mev", "kmu", "amf", "scatters"],
            default=["none"],
            nargs="+")
    parser.add_argument("-o", "--outfile", required=False,
            default=None)
    parser.add_argument("-r", "--random",
            default=False,
            action="store_true")
    parser.add_argument("--write-results-to-disk", required=False,
            default=False,
            action="store_true")
    parser.add_argument("--parallel", required=False,
            default=False,
            action="store_true")
    return parser


def main(args=None):
    parser = getMergerParser()
    args = parser.parse_args(args)
    args = vars(args)
    writtenFiles = run(**args)
    # print(writtenFiles)
    return writtenFiles



if __name__ == "__main__":
    main()
