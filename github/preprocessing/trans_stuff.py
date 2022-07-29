def copyRequisiteFilesAndFolders(requisite_files=None, verbose=False):
    from pathlib import Path
    import tempfile
    import sys
    import shutil
    # https://docs.python.org/3/library/tempfile.html
    if not isinstance(requisite_files, list):
        raise("requisite_files is not a list... is a {}".format(type(requisite_files)) )
    # Create a temporary directory just incase...
    # docs.python.org/3/library/tempfile.html
    tempDir = tempfile.TemporaryDirectory()
    tfiles = 0
    for item in requisite_files:
        p = Path(item)
        if p.is_dir():
            if verbose:
                print("Adding {}".format(str(p)))
            # Add to PYTHONPATH
            sys.path.append(str(p))
        elif p.is_file():
            tfiles += 1
            shutil.copyfile(p, "{}/{}".format(tempDir.name, p.name))
            if verbose:
                print("{} -> {}/{}".format(str(p), tempDir.name, p.name))

    # If there are no copied files, delete the tempdir, otherwise add the dir
    # to the PYTHONPATH
    if tfiles == 0:
        tempDir.cleanup()
        tempDir = None
        if verbose:
            print("No temp files to save, deleting {}".format(tempDir.name))
    else:
        sys.path.append(tempDir.name)
        if verbose:
            print("Adding {}".format(tempDir.name))
    return tempDir

def getTransformer(custom_transformer=None, state="save", verbose=False):
    import pathlib
    if state == "save":
        # Import the modules used __import__()
        transformer = custom_transformer.split(".")
        transformer = [".".join(transformer[:-1]), transformer[-1]]
        baseImport = __import__(transformer[0], globals(), locals(), [transformer[1]], 0)
        transformer = getattr(baseImport, transformer[1])
        transformer = transformer()
    elif state == "load":
        if not pathlib.Path(custom_transformer).is_file():
            raise("Custom transformer is not a file?: {}".format(custom_transformer))
        # Load the pickle file
        import pickle
        with open(custom_transformer, "rb") as tfile:
            transformer = pickle.load(tfile)
    else:
        raise("State other than load or save specified! state is {}".format(state))

    if verbose:
        print("Transformer loaded: {}".format(transformer))
    return transformer

def main(infile=None,
        state=None,
        custom_transformer=None,
        transformer_outfile=None,
        requisite_files=None,
        prefix=None,
        outfile=None,
        verbose=False
        ):
    import pandas
    import pathlib
    # Adjust path and copy files around
    if not requisite_files is None:
        tdir = copyRequisiteFilesAndFolders(requisite_files=requisite_files, verbose=verbose)
    # Load transformer in
    if not state is None:
        transformer = getTransformer(custom_transformer, state=state, verbose=verbose)
    elif state is None:
        raise("State is None? Catastrophic sadness!")
    # Load data in
    if infile is None:
        raise("infile is None? Something has gone wrong...")
    pInfile = pathlib.Path(infile)
    if pInfile.is_file():
        inframe = pandas.read_csv(infile)
    # Transform data
    if state == "save":
        outframe = transformer.fit_transform(inframe)
        # Pickle transformer out
        if transformer_outfile is None:
            if outfile is None:
                transformer_outfile = "{}_{}.pickle".format(
                    pInfile.stem, custom_transformer.split(".")[-1])
            # else:
                # pOutfile = pathlib.Path(outfile)
                # transformer_outfile = "{}/{}_{}.pickle".format(
                    # pOutfile.parent, pOutfile.stem, custom_transformer.split(".")[-1])
        # Generate transformer name
        if prefix is not None:
            transformer_outfile = "{}/{}".format(prefix, transformer_outfile)
        import pickle
        with open(transformer_outfile, "wb") as tout:
            pickle.dump(transformer, tout)
    elif state == "load":
        outframe = transformer.transform(inframe)
    # Save resulting data
    if prefix is not None:
        base = prefix
    else:
        base = ""
    if outfile is None:
        outframe.to_csv(base + "{}_{}.csv".format(pInfile.stem, "transformed"),
                index=False)
    else:
        outframe.to_csv(base + outfile, index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="""Transforms data
    given a transformer import syntax.""")
    parser.add_argument("-i", "--infile", required=True, default=None,
            help="""The infile to be processed. 
            The infile is assumed to be a CSV. 
            If it is not a CSV then... this script will not work.
            It does not need to have a CSV extension but
            the Python pathlib.Path class will be used to chop off
            the extension.
            This may cause unintended consequences if you have multiple '.'
            in the filename. 
            See pathlib.Path for more details.
            """)
    parser.add_argument("-p", "--prefix", required=False, default=None
            )
    parser.add_argument("-s", "--state", required=True, default="save",
            choices=["load", "save"],
            help="""Either you are LOADing a transformer
            to transform the data or you are fitting a transformer
            to your data to be SAVEd later.
            """)
    parser.add_argument("-ct", "--custom-transformer", required=True,
            help="""
            This allows the usage of any transformer.
            Simply provide importation style syntax like:
            'sklearn.preprocessing.StandardScaler'.
            An attempt will be made to import 'StandardScaler'
            from 'sklearn.preprocessing' .
            If you are using a custom transformer then you will 
            need to use the --requisite-files flag to 
            point out which non-builtin files are required to use your
            transformer. See --requisite-files for more information. 
            Your custom transformer will be pickled to disk after it has
            been fit to your data.
            The filename will be: 'dataname_transformername.pickle'
            unless you use --transformer-outfile to specify the name.
            If you were to use 'sklearn.preprocessing.StandardScaler'
            with 'samepleData.csv' then the transformer would be
            name 'sampleData_StandardScaler.pickle'.
            Your transformer must have a fit, transform, and a fit_transform.
            Keep in mind that your ENTIRE data file will be passed as
            a 'pandas.DataFrame' to your transformer.
            The result of the fit_transform/transform method will be saved
            to disk.
            """)
    parser.add_argument("-rf", "--requisite-files", required=False, default=None,
            nargs="+",
            metavar="FILE/DIR",
            help="""This flag is only required if you are using
            a transformer that is not an importable builtin.
            We use the __import__() function to pull in the
            class given a string like 'library.class'.

            In order to import non-standard classes in this manner, you will
            need to provide the absolute path and filename of
            the transformer.
            You will need to provide the absolute path and filename of
            any non-builtin files that the custom transformer imports from
            or requires for function.
            A special directory will be created in $TEMPDIR
            where COPIES of these specified requisite files are stored.
            This special directory will be temporarily added to your PYTHONPATH
            so that Python can automatically see them and pull what it needs
            from them.

            You can also provide a directory. 
            This WHOLE DIRECTORY will be added, temporarily, to your PYTHONPATH.
            The contents of this directory will not be touched.
            
            """)
    parser.add_argument("-to", "--transformer-outfile", required=False,
            default=None,
            help="""
            The is the filename that your transformer will be saved with.
            If your state is set to 'load' this option does nothing.
            If this option is not used then the transformer will be called
            'inputDatafileName_TransformerName.pickle'.
            If you were to use 'sklearn.preprocessing.StandardScaler'
            with 'samepleData.csv' then the transformer would be
            name 'sampleData_StandardScaler.pickle'.

            """)
    parser.add_argument("-o", "--outfile", required=False, default=None,
            help="""
            This is the filename where the transformed data will go.""")
    parser.add_argument("-v", "--verbose", required=False, default=False,
            action="store_true",
            help="""
            Makes a lot of noise!
            """)
    args = parser.parse_args()
    args = vars(args)
    main(**args)

