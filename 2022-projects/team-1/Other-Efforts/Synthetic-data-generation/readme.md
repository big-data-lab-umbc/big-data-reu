# Installation:
If you put this anywhere on Taki, utils/paths.py has global paths to some datasets already created
Otherwise, you'll have to create your own datasets and modify utils/paths.DATASET_PATHS

the shell scripts have global paths to a logging directory for the slurm outputs which you'll want to change

# General structure / vocab
"dataset names" are keywords pointing to directories containing datasets. They are defined in utils/paths.DATASET_NAMES
"datasets" are unorganized folders of images

"label sets" are csv files in data_loaders/labels/ with the naming scheme {label_set_name}_train, {label_set_name}_val, {label_set_name}_test, {label_set_name}_unlabeled

the transfer (and baseline) models take a dataset name and a label set and load images from the dataset directory according to the train, val, test labels
the localization model takes two dataset names, one containing augmented images to serve as validation and test sets and one containing unaugmented data to serve as the basis for generating the train set. The label set indicates which augmented images are test and which are validation and which unagumented images are to be used for training.

# Synthetic data generation:
Use train_localization_model.sh and train_localization_transfer_model.sh
The former takes args: image_dimension, n_epochs, gravity_wave_generation_keyword
The latter: image_dimension, n_epochs, gravity_wave_generation_keyword, n_epochs_the_localization_model_was_trained

The latter identifies the saved model from the former by deducing where my logging system saved it from gravity_wave_generation_keyword and n_epochs_the_localization_model_was_trained

# Using a different gravity wave generation method
Create a method analogous to data_loaders/gravity_wave_generation/generate_gravity_wave_simple.generateWavePattern
Then modify the parameters at the top of create_synthetic_dataset to use your wavePatternGeneration method and use it to create a new dataset and new labelset
    (don't forget to define your chosen dataset name in utils/paths.DATASET_PATHS first)
Then add a keyword identifying that wave pattern to utils/paths.MODE_DICTIONARY
And run train_localization_model.sh with that keyword

# Usage:
training logs are generated in the folder defined by utils/paths.USER_LOGS
these include a model summary, the weights of the best model over the training period, the weights of the final model, and a csv tracking the model's metrics and loss over time.
There is also a csv called model_results.csv which tracks the metrics of the best weights for every model trained

    bash train_localization_model.sh 256 300 simple
and then, once that finishes,
    bash train_localization_transfer_model.sh 256 300 simple 300
and
    bash train_localization_baseline.sh 256 300 simple
will produce the results I used in the technical report (use analysis/evaluate_transfer_model.ipynb for F-scores).

# Modifying the pipeline:
New data
    a) create a label csv files for train, val, test, and optionally unlabeled
    b) add the data to taki and the path to the data to utils/paths.DATASET_PATHS
    c) run the hdf5toPNG with the paths set appropriately (in preprocessing)
    d) optionally modify analysis/test_dataloaders.ipynb to make sure the dataset loaders are handling your new dataset correctly