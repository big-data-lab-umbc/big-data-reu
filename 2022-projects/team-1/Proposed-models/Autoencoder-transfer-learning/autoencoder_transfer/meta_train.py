### USAGE 
# python3 metatrain.py model_name dataset num_epochs run_date
# e.g. python3 metatrain.py basic_cnn labeled_data 2 0
## but actually invoke this through meta_slurm.slurm


# import os
# os.chdir("../") 

# from sys import argv


# if len(argv) >= 4:
#     early_stopping = bool(argv[3])


import tensorflow as tf

from paths import getTrainingLogDir
from basic_autoencoder_transfer import *
from seraj_filtered_data import *

# name_root = argv[0].split("/")[1][:-3]
epochs = 100

batch_size = 32

directory = '/home/kchen/reu2022_team1/research/autoencoder'

# NAME = "{}-{}-{}".format(model_name, dataset, epochs)

# log_dir = getTrainingLogDir(NAME, timestamp)
if batch_size is not None:
    train, val, test = getData()
else:
    train, val, test = getData()

model = getModel()

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=15, min_lr= 1e-12)

h = model.fit(
    train,
    steps_per_epoch=len(train),
    epochs=epochs,
    validation_data=val,
    validation_steps=len(val),
    verbose=1,
    callbacks=[
        tf.keras.callbacks.CSVLogger(directory + "/metrics.csv", separator=",", append=False),
        reduce_lr,
        tf.keras.callbacks.ModelCheckpoint(
            filepath=directory + "/best_model/", 
            save_freq="epoch",
            monitor='val_loss',
            mode='min',
            save_best_only=True,
        ), #,  save the best performing model
        # tf.keras.callbacks.EarlyStopping(
            # patience=10,
            # monitor="val_loss",
            # mode="min")
    ]
)

evaluateModel( model, val, directory )

stats = model.evaluate( val )
if len( model.metrics_names ) == 1:
    stats = [stats]

# write the final validation accuracy and loss to a csv file    
with open( directory + "/model_results.csv", "a") as fn:
    fn.write("autoencoder_refining: " + ", ".join( model.metrics_names ) + " extra dense layers " + ", ".join( [str(x) for x in stats] ) + "\n" )

val_pred = model.predict(val)
print(val_pred) # prints validation generator predictions

# get confusion matrix
import sklearn.metrics as metrics
import numpy as np
true_classes = val.classes
class_labels = list(val.class_indices.keys())
pred = np.round(val_pred)
confusion_matrix = metrics.confusion_matrix(y_true=true_classes, y_pred=pred)
print(confusion_matrix)

# get test accuracy
test_acc = model.evaluate(test)
print(test_acc)

# model.save(log_dir + "final_model/")
