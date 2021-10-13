
def main(data_type, infile, class_file=None, skip_class=False, outfile=None, rfilter=False):
    from pandas import read_csv
    polf_header = [
        'e1', 'x1', 'y1', 'z1',
        'e2', 'x2', 'y2', 'z2',
        'e3', 'x3', 'y3', 'z3']
    # Read in the scatter data
    oFrame = read_csv(infile, header=None,  names=polf_header)
    if class_file is not None:
        # Read in the identifying file
        tFrame = read_csv(class_file, header=None, names=["pClass"], dtype=int)
        # Merge
        oFrame['pClass'] = tFrame
        oFrame['class'] = 0 # zeros(oFrame.shape[0]).reshape((-1,1))
        # Assign a regular class
        if data_type == "triples" or data_type == "dtot" or data_type == "false":
            oFrame.loc[oFrame['pClass'] == 1, 'class'] = 14
            oFrame.loc[oFrame['pClass'] == 2, 'class'] = 8
            oFrame.loc[oFrame['pClass'] == 3, 'class'] = 12
        if data_type == "doubles":
            oFrame.loc[oFrame['pClass'] == 0, 'class'] = 6
            oFrame.loc[oFrame['pClass'] == 1, 'class'] = 14
            # This is incorrect I guess?
            # oFrame.loc[oFrame['pClass'] == 1, 'class'] = 6
            # oFrame.loc[oFrame['pClass'] == 2, 'class'] = 7
        # Remove any non pure classes
        if rfilter and data_type == "triples":
            oFrame = oFrame.loc[(oFrame['class'] == 0), :]

        if rfilter and data_type == "doubles":
            oFrame = oFrame.loc[(oFrame['class'] == 6), :]

        if rfilter and data_type == "false":
            oFrame = oFrame.loc[(oFrame['class'] == 14), :]

        if rfilter and data_type == "dtot":
            oFrame = oFrame.loc[(oFrame['class'] == 8) | (oFrame['class'] == 12), :]
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
            

    # Filter is requested
    r = polf_header[::]
    if not skip_class and not data_type == "singles":
        r.append("class")
    if outfile is None:
        from pathlib import Path
        outfile = Path(infile).stem
        outfile = "{}_converted.csv".format(outfile)
        # print(outfile)
    print(outfile)
    oFrame[r].to_csv(outfile, index=False)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Converts Polf's data into Maggi's format")
    parser.add_argument("-t", "--data-type", required=False, default=None,
            choices=["triples", "doubles", "singles", "dtot", "false"],
            # nargs=1,
            help="""Expects the type of data it should be correcting.
            Options: triples, doubles, singles. 
            Now that DtoT events will be in the triples file.""")
    parser.add_argument("-i", "--infile", required=True, default=None,
            # nargs=1,
            help="""The input file from polf. 
            We assume that the input file has no header. 
            If it does have a header this program will not work.""")
    parser.add_argument("-c", "--class-file",  required=False, default=None,
            # nargs=1,
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
    parser.add_argument("-o", "--outfile", required=False, default=None,
            # nargs=1,
            help="""Set the output name for the converted data file.
            If no name is provided will be the base filename with no extension
            and a _converted.csv appended to the end.""")
    args = parser.parse_args()
    # print(args.model)
    args = vars(args)
    # print(args)
    main(**args)

