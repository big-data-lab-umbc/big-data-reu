import numpy as np
from PIL import Image

def augmentImage(img, dim, augmentation_fnc):
    overlay, coords = augmentation_fnc(img, dim)
    aug = (overlay.reshape((dim, dim, 1)) + img.reshape((dim, dim, -1)))
    return overlay, ((aug - np.min(aug)) / (np.max(aug)-np.min(aug)) * 255).clip(0, 255), coords

def grabImage(path, dim):
    return np.array( Image.open( path ) ).astype(np.float64).reshape((dim,dim,-1))

### edited implementation of ImageDataGenerator().flow_from_dataframe()._get_batches_of_transformed_samples
from tensorflow.keras.preprocessing.image import (array_to_img, img_to_array, load_img)
def custom_get_batches(augmentation_fnc):
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
            _, x, new_label = augmentImage(x, len(x), augmentation_fnc) #### THIS IS THE SECOND THING I CHANGED
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
    return _custom_get_batches