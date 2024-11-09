from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import numpy as np
import pandas as pd
import argparse
import seaborn as sn
import matplotlib.pyplot as plt
# import sys
# sys.path.append('../')
# from train import PROJECT_BASE_PATH


PROJECT_BASE_PATH = '/nfs/rs/cybertrn/reu2024/team2/base/'  # need to import this from train


def get_args():
    parser = argparse.ArgumentParser(description='UMBC-HPCF PGML Team v0.0')
    parser.add_argument('-o', '--options', type=str, required=False, help='b for both, p for just pred, t for just train')
    parser.add_argument('-p', '--pred_path', type=str, required=False, help='directory you want to work out of that contains predict results relative to base/')
    parser.add_argument('-t', '--train_path', type=str, required=False, help='directory you want to work out of that contains train results relative to base/')
    args = parser.parse_args()
    return args


def plot_train(pth):
    # read in the data
    df = pd.read_csv(PROJECT_BASE_PATH+str(pth)+'metrics.csv')

    fig, ax = plt.subplots(2, figsize=(20, 10))
    title = f'max train acc: {np.max(df["train_acc_epoch"])}, max val acc: {np.max(df["valid_acc_epoch"])}'
    fig.suptitle(title)
    train_acc = df[["epoch","train_acc_epoch"]].dropna()
    val_acc = df[["epoch","valid_acc_epoch"]].dropna()
    train_loss = df[["epoch","train_loss"]].dropna()
    val_loss = df[["epoch","valid_loss"]].dropna()

    ax[0].plot(train_acc["epoch"],train_acc["train_acc_epoch"], label='train_acc')
    ax[0].plot(val_acc["epoch"],val_acc["valid_acc_epoch"], label='val_acc')
    ax[0].set(xlabel='epoch', ylabel='acc', title='accuracy')

    ax[1].plot(train_loss["epoch"],train_loss["train_loss"], label='train_loss')
    ax[1].plot(val_loss["epoch"],val_loss["valid_loss"], label='val_loss')
    ax[1].set(xlabel='epoch', ylabel='loss', title='loss')

    fig.legend()
    plt.savefig(PROJECT_BASE_PATH+str(pth)+'train.png')


def plot_pred(pth):
    # read in the data
    pred = np.load(PROJECT_BASE_PATH+str(pth)+'y_pred.npy')
    truth = np.load(PROJECT_BASE_PATH+str(pth)+'y_truth.npy')

    print('Accuracy: %.3f' % accuracy_score(truth, pred))

    # constant for classes
    classes = np.arange(0, 13, 1)

    # Build confusion matrix
    cf_matrix = confusion_matrix(truth, pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    plt.suptitle('Confusion Matrix Accuracy: %.3f' % accuracy_score(truth, pred))
    ticks = ["123", "132", "213", "231", "312", "321", "124", "214", "134", "314", "234", "324", "444"]
    sn.heatmap(df_cm, annot=True, xticklabels=ticks, yticklabels=ticks)
    #sn.heatmap(df_cm, annot=True)
    plt.savefig(PROJECT_BASE_PATH+str(pth)+'/cfsn.png')

    # save text file with classification report
    output = open(PROJECT_BASE_PATH+str(pth)+'scoring.txt', 'w')
    output.write('Accuracy: %.3f\n' % accuracy_score(truth, pred))
    # output.write('ROC AUC: %.3f\n' % roc_auc_score(np.expand_dims(truth, axis=1), np.expand_dims(pred, axis=1), multi_class='ovr'))
    output.write(classification_report(y_true=truth, y_pred=pred))
    output.close()


def main(args):
    if args.options == 'p' or args.options == 'b':
        plot_pred(args.pred_path)
    if args.options == 't' or args.options == 'b':
        plot_train(args.train_path)


if __name__ == "__main__":
    args = get_args()
    main(args)