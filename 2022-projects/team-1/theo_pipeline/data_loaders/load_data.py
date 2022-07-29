from data_loaders.utils import _rawTrain, _getLabels
from utils.paths import getDataDirectory

# load data in the various formats required by our models
# dataset_name is the keyword corresponding to the target dataset
#   (see utils.paths.DATASET_PATHS)
# optionally followed by a set of modifiers from [rgb, no_normalize, augment, no_rescale, no_shuffle]
# in the format keyword-modifier1-modifier2
# e.g. preprocessed_fft-rgb-no_shuffle
def getData(data_mode, dataset_name, label_set, dim=256):
    pieces = dataset_name.split("-")
    dataset_name = pieces[0]

    normalize=True
    augment=False
    rgb=False
    rescale=True
    batch_size=32
    shuffle=True

    if "no_normalize" in pieces: normalize = False
    if "augment" in pieces: augment = True
    if "rgb" in pieces: rgb = True
    if "no_rescale" in pieces: rescale = False
    if "no_shuffle" in pieces: shuffle = False
    # to do: batch size

    # for transfer learning purposes, we always use the same images to fit our normalization parameters
    if normalize:
        data_path = getDataDirectory("fft_data_{}".format(dim))
        train_data = _getLabels("synthetic_2018", "train")
        rt = _rawTrain(train_data, data_path, rgb, dim=dim)
    else: rt = None

    if data_mode == "labeled":
        from data_loaders import load_labeled_data
        return load_labeled_data._getData(dataset_name, label_set, normalize, augment, rgb, rescale, batch_size, shuffle, rt=rt, dim=dim)
    elif data_mode == "autoencoder":
        from data_loaders import load_autoencoder_data
        return load_autoencoder_data._getData(dataset_name, label_set, normalize, augment, rgb, rescale, batch_size, shuffle, rt=rt, dim=dim)
    elif data_mode == "coords":
        from data_loaders import load_localization_data
        return load_localization_data._getData(dataset_name, "synthetic_data_{}".format(dim), label_set, normalize, augment, rgb, rescale, batch_size, shuffle, rt=rt, dim=dim)