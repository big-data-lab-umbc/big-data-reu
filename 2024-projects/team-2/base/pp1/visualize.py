import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(description='UMBC-HPCF PGML Team v0.0')
    parser.add_argument('-p', '--path', type=str, required=False, help='data file you want to visualize relative to base/pp1/data/')
    args = parser.parse_args()
    return args


def write_report(df, path):
     # save text file with classification report
    output = open(path, 'w')
    output.write("df.describe: ")
    output.write(df.describe().to_string())
    output.write("\n")
    output.close()


def plot_hists(df, path):
    df.hist(figsize=(30, 30))
    plt.savefig(path)


def main(args):
    # read in data and create output directory
    df = pd.read_csv('data/'+args.path)
    dir = 'visualizations/'+str(args.path)[:-4] + '/'
    print('WARNING: '+dir+' already exists') if os.path.isdir(dir) else os.mkdir(dir)  # make directory to save visualizations inside

    write_report(df, dir+'report.txt')
    plot_hists(df, dir+'hists.png')
    

if __name__ == "__main__":
    args = get_args()
    main(args)