from tensorflow.keras.preprocessing.image import ImageDataGenerator

def _rawTrain(filepath, rgb):
    train_datagen = ImageDataGenerator(
        # rescale=1./255
    )

    color_mode = "rgb" if rgb else "grayscale"
    channels = 3 if rgb else 1
    train = train_datagen.flow_from_directory(
        filepath + "train/",
        target_size=(256, 256),
        batch_size=1,
        color_mode = color_mode,
        class_mode='binary',
        shuffle=False
    )
    rt = []
    for _ in range(len(train)):
        rt.append( train.next()[0].reshape( (256,256,channels) ) )
    return rt

# each train, val, test generator should return tuples of inputs and labels
def _getData(filepath, normalize=True, augment=True, rgb=False, batch_size=32, shuffle=True):
    if not shuffle:
        augment = False
    args = {
        "rescale": 1./255,
    }
    if augment:
        aug = {
            "shear_range": 0.2,
            "zoom_range": 0.2,
            "horizontal_flip": True,
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
        rt = _rawTrain(filepath, rgb)
        train_datagen.fit(rt)

    test_args = {
        "rescale": 1./255,
    }
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

    color_mode = "rgb" if rgb else "grayscale"
    train = train_datagen.flow_from_directory(
        filepath+'train',
        target_size=(256, 256),
        batch_size=batch_size,
        color_mode = color_mode,
        class_mode='binary',
        shuffle=shuffle
    )

    val = test_datagen.flow_from_directory(
        filepath+'validation',
        target_size=(256, 256),
        batch_size=batch_size,
        color_mode = color_mode,
        class_mode='binary',
        shuffle=False
    )

    test = test_datagen.flow_from_directory(
        filepath+'test',
        target_size=(256, 256),
        batch_size=1,
        color_mode = color_mode,
        class_mode='binary',
        shuffle=False
    )
    return train, val, test
