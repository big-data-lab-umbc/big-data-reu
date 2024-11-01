from pathlib import Path
import tempfile
import sys
import shutil
import pandas as pd
import pathlib
import os
from sklearn.model_selection import train_test_split
import numpy as np
import argparse


'''TRANSPLANT: these methods are from barajas and are untouched'''
def copyRequisiteFilesAndFolders(requisite_files=None, verbose=False):
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
'''END OF TRANSPLANT'''


def save_files(data_id, X_train_piped, X_test_piped, y_train_piped, y_test_piped):
    # make directories to save data inside
    data_dir = 'data/' + data_id + '/'
    train_dir = data_dir+'train/'
    test_dir = data_dir+'test/'
    print('WARNING: '+data_dir+' already exists') if os.path.isdir(data_dir) else os.mkdir(data_dir)
    print('WARNING: '+train_dir+' already exists') if os.path.isdir(train_dir) else os.mkdir(train_dir)
    print('WARNING: '+test_dir+' already exists') if os.path.isdir(test_dir) else os.mkdir(test_dir)

    # save the data
    pd.DataFrame(X_train_piped).to_csv('data/'+str(args.data_id)+'/train/X.csv', index=None)
    np.save('data/'+str(args.data_id)+'/train/X.npy', X_train_piped)
    np.save('data/'+str(args.data_id)+'/train/y.npy', y_train_piped)
    np.save('data/'+str(args.data_id)+'/test/X.npy', X_test_piped)
    np.save('data/'+str(args.data_id)+'/test/y.npy', y_test_piped)
    # also save as txt for readabilitiy, these files shouldn't be used
    np.savetxt('data/'+str(args.data_id)+'/train/X.txt', X_train_piped)
    np.savetxt('data/'+str(args.data_id)+'/train/y.txt', y_train_piped)
    np.savetxt('data/'+str(args.data_id)+'/test/X.txt', X_test_piped)
    np.savetxt('data/'+str(args.data_id)+'/test/y.txt', y_test_piped)


def split(split, path):
    # read in the data
    df = pd.read_csv(path)

    # split the data
    y = df['class']
    X = df.drop('class', axis=1)

    return train_test_split(X, y, test_size=split, stratify=y)


def get_args():
    parser = argparse.ArgumentParser(description='UMBC-HPCF PGML Team v0.0')
    parser.add_argument('-i', '--infile', type=str, help='Path to data file relative to pp1/data/')
    parser.add_argument('-d', '--data_id', type=str, help='Post-pp2 data id')
    parser.add_argument('-s', '--split', type=float, required=False, default=0.1, help='Test split percentage')
    args = parser.parse_args()
    return args


def main(args):
    X_train, X_test, y_train, y_test = split(args.split, args.infile)
    # for this pp, it works best to have these joined, this is hacky but oh well...
    train_df = (X_train.join(y_train)).reset_index()  # reset index for transformers, doesn't change anything JUST the pandas index...?
    test_df = (X_test.join(y_test)).reset_index()

    '''TRANSPLANT section: super hacky stuff in order to run barajas's preprocessing'''
    # do the requisite method
    dirname = os.path.dirname(__file__)
    req1 = os.path.join(dirname, 'transies.py')
    req2 = os.path.join(dirname, 'sampleTrans.py')
    copyRequisiteFilesAndFolders(requisite_files=[req1, req2])

    # Load transformer in 
    custom_transformer='sampleTrans.MyTriplesTransformer'
    transformer = getTransformer(custom_transformer)
   
    # transform the data
    X_train_piped = transformer.fit_transform(train_df)
    X_test_piped = transformer.transform(test_df)
    assert X_train_piped.isnull().sum().sum() == 0
    assert X_test_piped.isnull().sum().sum() == 0     
    new_cols = ['e1', 'x1', 'y1', 'z1', 'euc1', 'e2', 'x2', 'y2', 'z2', 'euc2', 'e3', 'x3', 'y3', 'z3', 'euc3', 'class']  
    X_train_piped = X_train_piped[new_cols]  # unnecessary but cleaner
    X_test_piped = X_test_piped[new_cols]

    y_train_piped = X_train_piped['class']
    X_train_piped = X_train_piped.drop('class', axis=1)
    y_test_piped = X_test_piped['class']
    X_test_piped = X_test_piped.drop('class', axis=1)
    '''END OF TRANSLPLANT'''

    # save the files
    save_files(args.data_id, X_train_piped, X_test_piped, y_train_piped, y_test_piped)


if __name__ == "__main__":
    args = get_args()
    main(args)

