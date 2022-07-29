Usage:
on taki use bash meta_train.sh
args are model_name dataset-dataset_args labelset num_epochs
e.g. bash meta_slurm.sh basic_autoencoder fft_data_256 2018 200

elsewhere use meta_train.py
python3 meta_train.py model_name dataset_name-dataset_args labels_to_use num_epochs
e.g. python3 meta_train.py basic_autoencoder fft_data_256-rgb 2018 200

Dataset args are used in data_loaders/load_data.py
dataset names are declared in utils/paths.py


This is a work in progress atm because I was rewriting it to make it easier to work on my project when I gave up on my project. I'll clean it up ... on the 26th probably, but the structure will remain the same.
The hdf5toPNG and fft processing are in preprocessing/hdf5_to_images.py