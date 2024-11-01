import pandas as pd
import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer


def get_args():
    parser = argparse.ArgumentParser(description='UMBC-HPCF PGML Team v0.0')
    parser.add_argument('-p', '--path', type=str, help='Path to data file relative to pp1/data_old/')
    parser.add_argument('-i', '--data_id', type=str, help='Post-pp2 data id')
    parser.add_argument('-s', '--split', type=float, required=False, default=0.1, help='Test split percentage')
    args = parser.parse_args()
    return args


def run_pipe(X_train, X_test):
    # pipeline: test and train must undergo IDENTICAL transformations
    # pipe = make_pipeline(StandardScaler(), Normalizer())
    # X_train_piped = pipe.fit_transform(X_train)
    # X_test_piped = pipe.fit_transform(X_test)
    # return X_train_piped, X_test_piped
    return X_train, X_test


def main(args):
    # read in the data
    df = pd.read_csv('../pp1/data_old/'+str(args.path))

    # split the data
    y = df['class']
    X = df.drop('class', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.split, stratify=y)

    # run the pipeline
    X_train_piped, X_test_piped = run_pipe(X_train, X_test)
    print("SHAPE: ", X_train_piped)

    # make directories to save data inside
    data_dir = 'data_old/'+str(args.data_id)
    train_dir = 'data_old/'+str(args.data_id)+'/train/'
    test_dir = 'data_old/'+str(args.data_id)+'/test/'
    print('WARNING: '+data_dir+' already exists') if os.path.isdir(data_dir) else os.mkdir(data_dir)
    print('WARNING: '+train_dir+' already exists') if os.path.isdir(train_dir) else os.mkdir(train_dir)
    print('WARNING: '+test_dir+' already exists') if os.path.isdir(test_dir) else os.mkdir(test_dir)

    # save the data
    np.save('data_old/'+str(args.data_id)+'/train/X.npy', X_train_piped)
    np.save('data_old/'+str(args.data_id)+'/train/y.npy', y_train)
    np.save('data_old/'+str(args.data_id)+'/test/X.npy', X_test_piped)
    np.save('data_old/'+str(args.data_id)+'/test/y.npy', y_test)
    # also save as txt for readabilitiy, these files shouldn't be used
    np.savetxt('data_old/'+str(args.data_id)+'/train/X.txt', X_train_piped)
    np.savetxt('data_old/'+str(args.data_id)+'/train/y.txt', y_train)
    np.savetxt('data_old/'+str(args.data_id)+'/test/X.txt', X_test_piped)
    np.savetxt('data_old/'+str(args.data_id)+'/test/y.txt', y_test)


if __name__ == "__main__":
    args = get_args()
    main(args)