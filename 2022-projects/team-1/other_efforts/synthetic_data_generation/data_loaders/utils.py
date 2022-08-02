import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def _getImageLoaders(normalize, augment, rgb, rescale, data_path, rt):
    args = {}

    if rescale:
        scale = {
            "rescale": 1./255,
        }
        args.update( scale )
    if augment:
        aug = {
            # "shear_range": 0.2,
            # "zoom_range": 0.2,
            "horizontal_flip": True,
            "vertical_flip": True,
        }
        args.update(aug)
    if normalize:
        norm = {
            "featurewise_center": True,
            "featurewise_std_normalization": True,
        }
        args.update(norm)
    train_datagen = ImageDataGenerator(
        **args
    )

    if normalize:
        train_datagen.fit(rt)

    test_args = {}
    if rescale:
        scale = {
            "rescale": 1./255,
        }
        test_args.update( scale )

    if normalize:
        norm = {
            "featurewise_center": True,
            "featurewise_std_normalization": True,
        }
        test_args.update(norm)

    test_datagen = ImageDataGenerator(
        **test_args
    )
    if normalize:
        test_datagen.fit(rt)

    return train_datagen, test_datagen

def _getLabels(label_set, subset="train"):
    data = pd.read_csv('data_loaders/labels/{}_{}.csv'.format( label_set, subset ))
    return data

def _rawTrain(training_data, data_path, rgb, dim=256):
    train_datagen = ImageDataGenerator()

    color_mode = "rgb" if rgb else "grayscale"
    channels = 3 if rgb else 1

    train = train_datagen.flow_from_dataframe(
        training_data, 
        directory=data_path,
        x_col="filename", y_col=["left", "top", "right", "bottom"],
        target_size=(dim, dim),
        batch_size=1,
        color_mode = color_mode,
        class_mode="raw",
        shuffle=False,
        follow_links=True,
    )

    ### they seriously don't support generators??
    # def terminatingGen():
    #     for _ in range(min(len(train))):
    #         img, _ = train.next()
    #         yield img.reshape((dim,dim,channels)) #np.full_like(img.reshape( (dim,dim,channels) ), np.mean(img) ) )
    # rt = terminatingGen()
    # return rt

    rt = []
    for _ in range(min(len(train), 1500)):
        img, _ = train.next()
        rt.append( np.full_like(img.reshape( (dim,dim,channels) ), np.mean(img) ) )
    return rt