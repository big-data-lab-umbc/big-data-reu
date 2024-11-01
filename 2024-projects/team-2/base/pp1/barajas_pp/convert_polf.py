def getBaseFilename(ofilename):
    import pathlib
    basename = pathlib.Path(ofilename).stem
    return basename


# def fixBadData(sFrame, top_data_type):
    # import pandas
    # import numpy
    # cols = {}
    # for col in sFrame.columns:
        # if (top_data_type == "doubles" and "3" not in col) or top_data_type=="triples":
            # cols[col] = pandas.to_numeric(sFrame[col], errors="coerce")    
    # # print(sFrame)
    # # print(cols)
    # newFrame = pandas.DataFrame(cols, dtype=float) 
    # newFrame = newFrame.dropna()
    # if top_data_type == "doubles":
        # for v in ['e3', 'x3', 'y3', 'z3']:
            # newFrame[v] = numpy.NaN
    # # print(newFrame)
    # return newFrame.reindex()


def fixBadData(sFrame, *args):
    import pandas
    import numpy
    cols = {}
    for col in sFrame.columns:
        cols[col] = pandas.to_numeric(sFrame[col], errors="coerce")    
    # print(sFrame)
    # print(cols)
    newFrame = pandas.DataFrame(cols, dtype=float) 
    I3 = ['e3', 'x3', 'y3', 'z3']
    
    # Will only drop columns.... hopefully
    newFrame = newFrame.dropna(how='all')
    # print(newFrame)
    I3Missing = sum([col not in newFrame.columns for col in I3]) == len(I3)
    newFrame = newFrame.dropna()
    if I3Missing == len(I3):
        # When handling doubles all I3 should be empty.
        print("I3 missing")
        for v in I3:
            newFrame[v] = numpy.NaN
    elif 0 < I3Missing and I3Missing < len(I3):
        # Stuff is just broken and idk how to fix it
        return DataFrame({}, columns=sFrame.columns)
    # print(newFrame)
    return newFrame.reindex()



def adjustFrame(infile=None, class_file=None, initial_energy_file=None, rfilter=False,
        no_energy_filter=False, top_data_type=None, polf_header=None, verbose=False):
    import pandas
    try:
        oFrame = pandas.read_csv(infile, header=None,  names=polf_header,
                dtype=float)
    except ValueError as e:
        import pathlib
        bname = pathlib.Path(infile).name
        print("{} | {}".format(bname, e))
        print("{} | Attempting to drop bad data...".format(bname))
        sFrame = pandas.read_csv(infile, header=None,  names=polf_header,
                dtype=str)
        oFrame = fixBadData(sFrame, top_data_type)
        print("{} | Bad Frame Shape   {}".format( bname, sFrame.shape))
        print("{} | New Frame Shape   {}".format(bname, oFrame.shape))
        print("{} | # of Skipped Rows {}".format(bname, sFrame.shape[0] - oFrame.shape[0]))
        
    if not initial_energy_file is None:
        eFrame = pandas.read_csv(initial_energy_file, header=None,  names=["E0"])
    else:
        eFrame = None

    if not class_file is None:
        # Read in the identifying file
        tFrame = pandas.read_csv(class_file, header=None, names=["pClass"], dtype=int)
        # Merge
        if not eFrame is None:
            oFrame['E0'] = eFrame

        oFrame['pClass'] = tFrame
        oFrame['class'] = 0 # zeros(oFrame.shape[0]).reshape((-1,1))

        # Assign a regular class
        if top_data_type == "triples":
            oFrame.loc[oFrame['pClass'] == 1, 'class'] = 14
            oFrame.loc[oFrame['pClass'] == 2, 'class'] = 8
            oFrame.loc[oFrame['pClass'] == 3, 'class'] = 12

        if top_data_type == "doubles":
            oFrame.loc[oFrame['pClass'] == 0, 'class'] = 6
            oFrame.loc[oFrame['pClass'] == 1, 'class'] = 14
            # This is incorrect I guess?
            # oFrame.loc[oFrame['pClass'] == 1, 'class'] = 6
            # oFrame.loc[oFrame['pClass'] == 2, 'class'] = 7



        # Remove and fix any non-pure classes
        if rfilter and top_data_type == "triples":

            # oFrame = oFrame.loc[(oFrame['class'] == 0), :]
            # oFrame = oFrame.loc[(oFrame['class'] == 6), :]
            # Correct DtoT events
            tFrame = oFrame.copy()
            I1 = ['e1', 'x1', 'y1', 'z1']
            I2 = ['e2', 'x2', 'y2', 'z2']
            I3 = ['e3', 'x3', 'y3', 'z3']
            # Convert class 12 (234) to class 8 (124)
            # Rebind first interaction
            oFrame.loc[(oFrame['class'] == 12), I1] = \
                    tFrame.loc[(tFrame['class'] == 12), I2].to_numpy()
            # Rebind second interaction
            oFrame.loc[(oFrame['class'] == 12), I2] = \
                    tFrame.loc[(tFrame['class'] == 12), I3].to_numpy()
            # Rebind third interaction
            oFrame.loc[(oFrame['class'] == 12), I3] = \
                    tFrame.loc[(tFrame['class'] == 12), I1].to_numpy()
            # Rebind class
            oFrame.loc[(oFrame['class'] == 12), 'class'] = 8

            # oFrame = oFrame.loc[(oFrame['class'] == 14), :]
            if not eFrame is None:
                oFrame.loc[(oFrame['class'] == 14), ["E0"]] = 0



        if      (not no_energy_filter) and \
                (not (eFrame is None)): 
            zeroEn = oFrame.loc[
                    (oFrame['E0'] == 0) &
                    (oFrame['class'] != 14), :].shape[0]
            # Select non-zero energies and false events
            oFrame = oFrame.loc[
                    (oFrame['E0'] != 0) | 
                    (oFrame['class'] == 14), :]
            # Zero out arbitrarily small values
            oFrame.loc[oFrame['E0'].abs() < 1e-6, 'E0'] = 0
            # Set Initial energy for false events to 0
            oFrame.loc[oFrame['class'] == 14 ,    'E0'] = 0
            # dE = DataFrame(columns=["dE"])
            # dE['dE'] = oFrame['E0'].subtract(oFrame['e1']).subtract(oFrame['e2']).subtract(oFrame['e2'])
            # oFrame = oFrame.loc[dE['dE'] >= 0, :]
            if verbose:
                print("Removed {} E0 == 0 entries".format(zeroEn))
                # print("Removed {} dE < 0".format(dE[dE['dE'] < 0].shape[0]))
    return oFrame

def padFileLists(infiles, class_files, initial_energy_files, prefix):
    if isinstance(prefix, str):
        prefixs = [prefix]
    if class_files is None:
        class_files = []
    if initial_energy_files is None:
        initial_energy_files = []
    if infiles is None:
        print("Infile is None?")
        exit(1)
    for i in range(len(infiles)):
        if len(class_files) < i+1:
            class_files.append(None)
        if len(initial_energy_files) < i+1:
            initial_energy_files.append(None)
        if len(prefixs) < i+1:
            prefixs.append(prefix)
    return infiles, class_files, initial_energy_files, prefixs

# We need to use this function with the passthroughWrapper
# This allows us to use the ** notation with processFile
def padFileListsAsDicts(
        infiles, class_files, initial_energy_files, prefix,
        rfilter, no_energy_filter, top_data_type, polf_header,
        skip_class,
        verbose):
    if initial_energy_files is None:
        initial_energy_files = []
    if class_files is None:
        class_files = []
    prefixs = prefix
    listOfDicts = []
    for i in range(len(infiles)):
        infile = infiles[i]
        
        if len(class_files) < i+1:
            class_file = None
        else:
            class_file = class_files[i]

        if len(initial_energy_files) < i+1:
            initial_energy_file = None
        else:
            initial_energy_file = initial_energy_files[i]

        if isinstance(prefix, str):
            p = prefix
        elif isinstance(prefix, list) and len(prefix) < i+1:
            p = "./"

        listOfDicts.append(
            {
                'infile': infile,
                'class_file': class_file,
                'initial_energy_file': initial_energy_file,
                'rfilter': rfilter,
                'prefix': p,
                'no_energy_filter': no_energy_filter,
                'top_data_type': top_data_type,
                'polf_header': polf_header,
                'skip_class': skip_class,
                'verbose': verbose,
            }       
        )
    return listOfDicts

# This allows us to take in a dictionary and use it as if it was
# a collection of input arguments to a function.
# This is what **var does.
def passthroughWrapper(params):
    writtenFiles = processFile(**params)
    return writtenFiles

# def processFile(infile, class_file, initial_energy_file, prefix):
def processFile(infile=None, class_file=None, initial_energy_file=None, rfilter=False,
        no_energy_filter=False, top_data_type=None, polf_header=None,
        prefix="./", skip_class=None, verbose=False):
    writtenFiles = []
    if verbose:
        print(infile)
        print("+", class_file)
        print("+", initial_energy_file)
    
    basename = getBaseFilename(infile)        

    oFrame = adjustFrame(infile=infile, class_file=class_file,
            initial_energy_file=initial_energy_file, rfilter=rfilter,
            no_energy_filter=no_energy_filter, top_data_type=top_data_type,
            polf_header=polf_header,
            verbose=verbose)
    if not class_file is None:# and not skip_class:
        sub_data_types = {
                "triples": [
                    ["triples",  0], 
                    ["dtot",     8],
                    ["false",   14]
                ],
                "doubles": [
                    ["doubles",  6], 
                    ["false",   14]
                ],
                "singles": [
                    ["singles", 14]
                ]
                }
        # print(sub_data_types[top_data_type])
        for data_type, nClass in sub_data_types[top_data_type]:
            # Filter is requested
            r = polf_header[::]
            if "E0" in oFrame.columns: 
                r.append("E0")
            if not skip_class and not data_type == "singles":
                r.append("class")
            outfile = "{}/{}_{}_converted.csv".format(
                    prefix,
                    basename,
                    data_type
                    )
            # print(outfile)
            writtenFiles.append(outfile)
            oFrame.loc[oFrame['class'] == nClass, r].to_csv(outfile, index=False)

    else:

        outfile = "{}/{}_noclass_converted.csv".format(
                prefix,
                basename,
                )
        writtenFiles.append(outfile)
        # print(outfile)

        oFrame[polf_header].to_csv(outfile, index=False)
    if verbose:
        print("----------")
    return writtenFiles


def run(infiles=None, top_data_type=None, class_files=None, skip_class=False, 
        rfilter=False, no_energy_filter=False, initial_energy=None,
        prefix=None,
        verbose=False):
    # print(infiles)
    polf_header = [
        'e1', 'x1', 'y1', 'z1',
        'e2', 'x2', 'y2', 'z2',
        'e3', 'x3', 'y3', 'z3']
    if prefix is None:
        prefix == "./"
    from pandas import read_csv, DataFrame
    # We use this generate a bunch of dictionary we will serve as inputs
    # the processFile functions.
    allDicts = padFileListsAsDicts(
        infiles=infiles, 
        class_files=class_files, 
        initial_energy_files=initial_energy, 
        prefix=prefix,
        rfilter=rfilter, 
        no_energy_filter=no_energy_filter, 
        top_data_type=top_data_type, 
        polf_header=polf_header,
        verbose=verbose, skip_class=skip_class)
    # print(allDicts)
    # Read in the scatter data
    # infile = infile[0]
    import multiprocessing
    import psutil
    CORES = psutil.cpu_count()
    # CORES = 1
    if verbose:
        print(CORES)

    with multiprocessing.Pool(CORES) as p:
        # We need to use the pass through wrapper like this because
        # Python multi-process does not handle lambdas/partials very 
        # well if at all.
        listsolists = p.map(passthroughWrapper, allDicts)
    writtenFiles = []
    for l in listsolists:
        writtenFiles.extend(l)
    return writtenFiles

def getConverterParser():
    import argparse
    parser = argparse.ArgumentParser(description="Converts Polf's data into Maggi's format")
    parser.add_argument("-t", "--top-data-type", required=False, default=None,
            choices=["triples", "doubles", "singles"],#, "dtot", "false"],
            # nargs=1,
            help="""Expects the type of data it should be correcting.
            Options: triples, doubles, singles. 
            DtoT and false events will be automatically saved to their
            own files.""")
    parser.add_argument("-p", "--prefix", required=False, default="./",
            help="""
            This is the directory where the files will be stored.
            Default is ./ .
            """)
    parser.add_argument("-i", "--infiles", required=True, default=None,
            # nargs=1,
            nargs="+",
            help="""The input file from polf. 
            We assume that the input file has no header. 
            If it does have a header this program will not work.""")
    parser.add_argument("-c", "--class-files",  required=False, default=None,
            # nargs=1,
            nargs="+",
            help="""This is the polf class file. Typically it will end in
            dType.txt or tType.txt. 
            If no class file is provided the new file will just contain a proper
            header.
            """)
    parser.add_argument("--skip-class", required=False, default=False,
            action="store_true",
            help="""When the class is provided, normally, the converted class
            is written to the output file.
            When this boolean is used, the converted class data is not written
            to the output file.
            This could be useful if you intend on filtering to only pure data
            and then using alternate preprocessing.
            """)
    parser.add_argument("-f", "--rfilter", required=False, default=False,
            action="store_true",
            help="""When provided will remove any 'non-true' aka misordered
            cases from the file before saving.""")
    parser.add_argument("-e0", "--initial-energy", required=False, default=None,
            # nargs=1,
            nargs="+",
            help="""When provided an initial energy file the converter
            will load in the initial energy file and add it as a column
            just before 'class' as 'E0'.
            If not file is provided nothing will changed.""")
    parser.add_argument("--no-energy-filter", required=False, default=False,
            action="store_true",
            help="""
            If you want to use initial energy then by default we 
            throw away events that have an initial energy of 0.
            If you want to keep the data that has an initial energy
            of 0 you MUST pass this flag!
            This flag has no impact when "data_type == 'false'" 
            and is ignored.
            """)
    parser.add_argument("-v", "--verbose", required=False, default=False,
            action="store_true",
            help="I make a lot of noise!")
    return parser

def main(args=None):
    parser = getConverterParser()
    args = parser.parse_args(args)
    args = vars(args)
    writtenFiles = run(**args)
    # print(writtenFiles)
    return writtenFiles


if __name__ == "__main__":
    main()
