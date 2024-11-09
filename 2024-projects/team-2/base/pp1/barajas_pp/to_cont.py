import pandas as pd


# method to change labels to 0-12
def to_cont(label):
    if label >= 8:
        return label - 2
    else:
        return label


def main(infile, outfile):
    df = pd.read_csv(infile)
    df['class'] = df['class'].apply(to_cont)
    # new_cols = ['e1', 'x1', 'y1', 'z1', 'e2', 'x2', 'y2', 'z2', 'e3', 'x3', 'y3', 'z3']
    # df = df[new_cols]
    df.to_csv(outfile, index=False)
    print("saved output file to ", outfile)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="makes the labels continuous")
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
    parser.add_argument("-o", "--outfile", required=False, default=None,
            help="""
            This is the filename where the transformed data will go.""")
    args = parser.parse_args()
    args = vars(args)
    main(**args)

