USER = "kchen" # CHANGE TO CURRENT USER
TEAM = "team1"

USER_ROOT = "/umbc/xfs1/cybertrn/users/{}/research/".format(USER)
TEAM_ROOT = "/umbc/xfs1/cybertrn/reu2022/{}/research/".format(TEAM)

# USER_LOGS = USER_ROOT + "logs/"

ORIGINAL_LABELED_GW = TEAM_ROOT + "data/"
LABELED_GW = TEAM_ROOT + "expanded_data/"
UNLABELED_GW = TEAM_ROOT + "expanded_data/unlabeled"
NEWLY_ORGANIZED_GW = TEAM_ROOT + "newly_organized_pngs/"
FILE_PATH_DF = TEAM_ROOT + "autoencoder/file_path_df.csv"
DIFFRANET = TEAM_ROOT + "diffranet/"
EXPANDED_FFT = TEAM_ROOT + 'fft-denoised-20220722T045526Z-001/fft-denoised/'

def getTrainingLogDir():
    fp = TEAM_ROOT + "autoencoder/"
    return fp

'''
def getTrainingLogDir(run_name="TEST", timestamp=None, make=True):
    import datetime
    from os import path, mkdir
    fp = getTrainingLogRoot(run_name)
    if not path.exists(fp):
        mkdir( fp )
    if timestamp is not None:
        fp = fp + "{}/".format(timestamp)
    else:
        fp = fp + "{}/".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    if not path.exists(fp):
        mkdir( fp )
    return fp
'''
