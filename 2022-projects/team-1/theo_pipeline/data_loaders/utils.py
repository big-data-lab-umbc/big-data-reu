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



### edit image generator to augment outputted images with a synthetic gravity wave and localization label
from numpy import sin, cos, pi, sqrt
import random
import math
import numpy as np
from PIL import Image

def fadeout(pos, total_length, m):
    if pos > total_length / 2:
        pos = total_length - pos
    
    D = max(m, total_length*0.4)
    return min( pos / D, 1 )

def generateWavePattern( img, dim ):
    dim = dim*1.
    ## random params...
    x, y = random.randint(int(0.05*dim), int(dim-0.05*dim)), random.randint(int(0.05*dim), int(dim-0.05*dim)) # where is the first pixel?
    period = random.randint(int(0.03*dim), int(0.2*dim)) # in pixels
    width = random.randint( int(max(0.1*dim, period*2)), int(min(0.6*dim, 5*period)) ) # also in pixels
    radius = random.randint(int(0.1*dim), int(10*dim)) # radius of circle whose circumference is being traced (pixels)
    length = random.randint(width, int(0.9*dim)) # along direction of wave
    init_theta = random.random() * 2 * pi # start where on circle
    amplitude = random.random() # How bright is the wave (in ... units)

    ## calculated values
    img_sigma = np.std(img[:,:,0])

    radians = length / radius
    max_t = radians + init_theta
    num_periods = width / period
    cx, cy = int(x - radius * cos(init_theta)), int(y - radius * sin(init_theta))
    
    # ds/dtheta = r, so never skip a pixel with
    dtheta = 1./(radius+width*period)# / 2

    overlay = np.zeros( (int(dim),int(dim)) )

    t = init_theta
    lx, ly = None, None

    # basic wave pattern
    A = [np.sin(j * 2 *np.pi / period) for j in range(width * period)]

    a, b, c, d = 1e6, 1e6, -1e6, -1e6
    off_x, off_y = 0, 0
    whiteout_c = 0
    total_whiteout = 0
    while t < max_t:
        x, y = cos(t)*radius, sin(t)*radius
        xd = x/sqrt(x**2+y**2)
        yd = y/sqrt(x**2+y**2)

        if x == lx and y == ly: continue
        
        r = random.random()
        if r < 0.02 and whiteout_c == 0:
            # off_x += 0.75 * period * x/sqrt(x**2+y**2)
            # off_y += 0.75 * period * x/sqrt(x**2+y**2)
            whiteout_c = int(radians / dtheta * 0.2 * (random.random()/2+0.5))
            total_whiteout = whiteout_c

        lxp, lyp = None, None
        s = 0
        while s < width * period:
            xp = (cx + x + s * xd + off_x)
            yp = (cy + y + s * yd + off_y)
            
            
            ixp, iyp = int(xp), int(yp)

            # to do blur / noise
            if ixp >= dim or iyp >= dim or ixp < 0 or iyp < 0:
                break
                
            if overlay[ixp, iyp] == 0:  
                overlay[ixp, iyp] +=  img_sigma * (1+amplitude) * A[int(s)] * fadeout(int(s), width, 0.02*dim) * fadeout(t-init_theta, radians, 0.02 * dim / radius) * (1 if whiteout_c == 0 else 1 - 2*fadeout(whiteout_c, total_whiteout, total_whiteout) )
            
            # advance to the next cell in our linear projection
            if x < 0:
                dx = math.floor( xp ) - xp
            else:
                dx = math.ceil( xp ) - xp
            if y < 0:
                dy = math.floor( yp ) - yp
            else:
                dy = math.ceil( yp ) - yp
            dx /= xd
            dy /= yd
            
            if dx < dy:
                s += dx + 1e-1
            else:
                s += dy + 1e-1

        if whiteout_c > 0:
            whiteout_c -= 1
        t += dtheta
        lx = x
        ly = y
        
    return overlay, (a/dim,b/dim,c/dim,d/dim)

def augmentImage(img, dim):
    overlay, coords = generateWavePattern(img, dim)
    aug = (overlay.reshape((dim, dim, 1)) + img.reshape((dim, dim, -1)))
    return overlay, ((aug - np.min(aug)) / (np.max(aug)-np.min(aug)) * 255).clip(0, 255), coords

def grabImg(path, dim):
    return np.array( Image.open( path ) ).astype(np.float64).reshape((dim,dim,-1))

from tensorflow.keras.preprocessing.image import (array_to_img, img_to_array, load_img)

### edited implementation of ImageDataGenerator().flow_from_dataframe()._get_batches_of_transformed_samples
def _custom_get_batches(self, index_array):
    batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=self.dtype)
    batch_y = np.zeros((len(batch_x), 4), dtype=self.dtype) #### ADDED THIS
    # build batch of image data
    # self.filepaths is dynamic, is better to call it once outside the loop
    filepaths = self.filepaths
    for i, j in enumerate(index_array):
        img = load_img(filepaths[j],
                       color_mode=self.color_mode,
                       target_size=self.target_size,
                       interpolation=self.interpolation)
        x = img_to_array(img, data_format=self.data_format)
        _, x, new_label = augmentImage(x, dim=len(x)) #### THIS IS THE SECOND THING I CHANGED
        # Pillow images should be closed after `load_img`,
        # but not PIL images.
        if hasattr(img, 'close'):
            img.close()
        if self.image_data_generator:
            params = self.image_data_generator.get_random_transform(x.shape)
            x = self.image_data_generator.apply_transform(x, params)
            x = self.image_data_generator.standardize(x)
        batch_x[i] = x
        batch_y[i] = new_label ### ALSO CHANGED
    #### ALSO ADDED THIS
    return batch_x, batch_y