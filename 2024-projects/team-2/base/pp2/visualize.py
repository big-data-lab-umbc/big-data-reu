import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description='UMBC-HPCF PGML Team v0.0')
    parser.add_argument('-p', '--path', type=str, required=False, help='data directory you want to visualize relative to base/pp2/data/, ends with /')
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
    df.hist(figsize=(30, 30), bins=100)
    plt.savefig(path)


def main(args):
    # read in data and create output directories
    train_X = pd.DataFrame(np.load('data/'+args.path+'train/X.npy'))
    test_X = pd.DataFrame(np.load('data/'+args.path+'test/X.npy'))  # can add y's later
    dir = 'visualizations/'+str(args.path)
    train_dir = dir + 'train/'
    test_dir = dir + 'test/'
    print('WARNING: '+dir+' already exists') if os.path.isdir(dir) else os.mkdir(dir)  # make directory to save visualizations inside
    print('WARNING: '+train_dir+' already exists') if os.path.isdir(train_dir) else os.mkdir(train_dir)  # make directory to save visualizations inside
    print('WARNING: '+test_dir+' already exists') if os.path.isdir(test_dir) else os.mkdir(test_dir)  # make directory to save visualizations inside

    # train
    write_report(train_X, train_dir+'report.txt')
    plot_hists(train_X, train_dir+'hists.png')

    # test
    write_report(test_X, test_dir+'report.txt')
    plot_hists(test_X, test_dir+'hists.png')
    

if __name__ == "__main__":
    args = get_args()
    main(args)