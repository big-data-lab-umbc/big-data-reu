### python3 meta_train.py model_name dataset_name-dataset_args labels_to_use num_epochs
### e.g. python3 meta_train.py basic_autoencoder fft_data_256-rgb 2018 200

import tensorflow as tf
import numpy as np
from sys import argv

model_name = argv[1]
dataset_name = argv[2]
label_set = argv[3]
epochs = int(argv[4])

timestamp = argv[-1] if len(argv) > 5 else None

exec('from models.{} import getModel, evaluateModel, getVersion, getDataType'.format(model_name))
from data_loaders.load_data import getData
from utils.paths import getTrainingLogDir, USER_LOGS
from models.utils import IOU_metric

version = getVersion()
NAME = "{}-{}/{}/{}/{}".format(model_name, version, dataset_name, label_set, epochs)
log_dir = getTrainingLogDir(NAME, timestamp)

model = getModel()
def printsummary(s):
    with open(log_dir + "README",'a') as f:
        print(s, file=f)
model.summary(print_fn=printsummary)

train, val, test = getData(getDataType(), dataset_name, label_set)
if train.class_mode == "binary":
    true_classes = np.array(train.classes)
    n0 = sum(true_classes == 0)
    n1 = sum(true_classes == 1)
    class_weights = {
        0: len(true_classes) / 2. / n0,
        1: len(true_classes) / 2. / n1,
    }
else: class_weights = None

print( "Starting fit" )
h = model.fit(
    train,
    steps_per_epoch=len(train),
    epochs=epochs,
    validation_data=val,
    validation_steps=len(val),
    class_weight=class_weights,
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
        # tf.keras.callbacks.EarlyStopping(
        #     patience=4,
        #     monitor="val_loss",
        #     mode="min"
        # ),
        # tf.keras.callbacks.ReduceLROnPlateau(
        #     monitor='val_loss', 
        #     factor=0.4,
        #     patience=2, 
        #     min_lr=1e-12,
        #     mode="min"
        # ),
    ]
)
model.save(log_dir + "final_model/")


# TODO load weights and put them in model...
best_model = getModel()
best_model.load_weights(log_dir+"best_model/")

# best_model = model 
evaluateModel( best_model, val, log_dir )

stats = best_model.evaluate( test )
if len( best_model.metrics_names ) == 1:
    stats = [stats]
    
with open( USER_LOGS + "git_repo/" + "model_results.csv", "a") as fn:
    fn.write( NAME + ": " + ", ".join( best_model.metrics_names ) + " | " + ", ".join( [str(x) for x in stats] ) + "\n" )