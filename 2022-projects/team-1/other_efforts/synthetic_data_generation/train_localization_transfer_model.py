### python3 meta_train.py model_name dataset_name-dataset_args labels_to_use num_epochs
### e.g. python3 meta_train.py basic_autoencoder fft_data_256-rgb 2018 200

import tensorflow as tf
import numpy as np
from sys import argv

from data_loaders.minimal_load_labeled_data import getData
from models.localization_inceptionv3_transfer import getModel, getVersion
from utils.paths import getTrainingLogDir, USER_LOGS, LOG_DIR_NAME, MODE_DICTIONARY

model_name = "localization_inceptionv3_transfer"

DIM = int(argv[1]) # positive integer [use 256]
epochs = int(argv[2]) # positive integer
synthetic_wave_type = argv[3] # "simple" or "default" [use "simple"]
pretraining_epochs = int(argv[4])

timestamp = argv[-1] if len(argv) > 5 else None # set by train.sh else unset

# use the original wave generation which wasn't diverse enough ("simple")
# or the new wave generation which is really slow and has lots of other problems ("default")
config = MODE_DICTIONARY[synthetic_wave_type]


version = getVersion()
NAME = "{}-{}/{}/{}/{}-{}".format(model_name, version, config["dataset_name"], config["transfer_label_set"], epochs, pretraining_epochs)
log_dir = getTrainingLogDir(NAME, timestamp)

model = getModel(n_epochs=pretraining_epochs, dataset_name=config["dataset_name"], label_set=config["label_set"])


## I find this helpful to remember what each run was and to verify that my model is set up the way I want it
def printsummary(s):
    with open(log_dir + "README",'a') as f:
        print(s, file=f)
model.summary(print_fn=printsummary)

train, val, test = getData( 
    config["dataset_name"], config["transfer_label_set"], DIM 
)

h = model.fit(
    train,
    steps_per_epoch=len(train),
    epochs=epochs,
    validation_data=val,
    validation_steps=len(val),
    verbose=0,
    callbacks=[
        tf.keras.callbacks.CSVLogger(log_dir + "metrics.csv", separator=",", append=False),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=log_dir+"best_model/", 
            save_freq="epoch",
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            save_weights_only=True
        ), # save the best performing model
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.1,
            patience=20, min_lr= 1e-15
        )
    ]
)
model.save(log_dir + "final_model/")

best_model = getModel(n_epochs=pretraining_epochs, dataset_name=config["dataset_name"], label_set=config["label_set"])
best_model.load_weights(log_dir+"best_model/")

train_stats = best_model.evaluate( train )
val_stats = best_model.evaluate( val )
test_stats = best_model.evaluate( test )
if len( best_model.metrics_names ) == 1:
    train_stats = [train_stats]
    val_stats = [val_stats]
    test_stats = [test_stats]

stats = train_stats + val_stats + test_stats
names = ["train_" + n for n in best_model.metrics_names] + ["val_" + n for n in best_model.metrics_names] + ["test_" + n for n in best_model.metrics_names]
    
with open( USER_LOGS + LOG_DIR_NAME + "/model_results.csv", "a") as fn:
    fn.write( NAME + ": " + ", ".join( names ) + " | " + ", ".join( [str(x) for x in stats] ) + "\n" )