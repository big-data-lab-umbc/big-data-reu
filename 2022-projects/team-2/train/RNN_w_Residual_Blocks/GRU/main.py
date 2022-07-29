from model_rnn import create_model
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import Callback as KCallback  
from datetime import datetime
from numpy import concatenate
##############################
# Do not change this file :) #
##############################

def getNormalization(normies=None):
    if not normies is None:
        # Get the importable function
        normies = normies.split(".")
        baseImport = __import__(normies[0], globals(), locals(), [normies[1]], 0)
        normies = getattr(baseImport, normies[1])
    return normies

def deltaToString(tme):
    sec = tme.total_seconds()
    hours = int(sec) // 60 // 60
    minutes = int(sec - hours* 60*60) // 60
    sec = sec - hours* 60*60 - minutes * 60
    return "{:02d}:{:02d}:{:05.2f}".format(hours, minutes, sec)

class TimeHistory(KCallback):
    def on_epoch_begin(self, epoch, logs):
        self.epoch_time_start = datetime.now()
    def on_epoch_end(self, epoch, logs):
        logs["epoch_time"] = deltaToString(datetime.now() - self.epoch_time_start)

def getInputClass(x,y, iclass):
    from numpy import array, where, argmax, isin
    index = {
    'triples': [0, 1, 2, 3, 4, 5],
    'doubles': [6, 7],
    'DtoT':    [8, 9, 10, 11, 12, 13],
    'false':   [14]}
    if isinstance(iclass, list):
        i = list(map(lambda x: index[x], iclass))
        valid_index = []
        for d in i:
            valid_index.extend(d)
        valid_index = array(valid_index)
    else:
        valid_index = array(index[iclass])
    t = argmax(y, axis=1).reshape(y.shape[0], 1)
    z1, z2 = where(isin(t, valid_index))
    x, y = x[z1], y[z1]
    return x, y

def getDevices():
    from tensorflow import config
    devices = [device for device in config.list_physical_devices() if "GPU" == device.device_type]
    devices = ["/gpu:{}".format(i) for i, device in enumerate(devices)]
    return devices

def set_tf_config_mpi():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    host = MPI.Get_processor_name()

    # Create the config dict
    import getPort
    prt = getPort.find_free_port()
    host =  "{}:{}".format(host, prt)
    tf_config = {}
    hosts = comm.allgather(host)
    hosts = [xhost for xhost in hosts]
    tf_config['cluster'] = {'worker': hosts}
    tf_config['task']    = {'type': 'worker', 'index': rank}
    print(rank, tf_config)

    # Dump into the environment for tensorflow to use
    # This is the suggested method on the docs for some reason
    from os import environ
    import json
    environ['TF_CONFIG'] = json.dumps(tf_config)

def createCallbacks(params, callbacks, rank, resume_training):
    callbacks.append(TimeHistory())
    if resume_training:
        csv = CSVLogger("training_log_{:02d}.csv".format(rank), append=True)
    else:
        csv = CSVLogger("training_log_{:02d}.csv".format(rank), append=False)
    callbacks.append(csv)
    checkpoints = ModelCheckpoint("checkpoints/model-"+"{epoch:05d}", 
            monitor='val_loss', period=64,verbose=0)
    callbacks.append(checkpoints)
    return callbacks

def removeDoubles(yold):
    from numpy import empty
    import numpy
    # Index 6 and 7 are the doubles to be removes
    ynew = empty((yold.shape[0],y.shape[1]-2))
    ynew[:,:6] = yold[:,:6]
    ynew[:,6:] = yold[:,8:]
    return ynew

def getInitialEpochsAndModelName(rank):
    from os import listdir
    modelName = None
    initial_epoch = 0
    items = listdir()
    # Check for an existing checkpoints folder
    if "checkpoints" in items:
        # Find the latest checkpoint
        checkpoints = listdir("checkpoints/")
        epochNumbers = list(map(lambda s: int(s.split("-")[1]), checkpoints))
        if len(epochNumbers) > 0:
            lastCheckpoint = max(epochNumbers)
            checkpointInd = epochNumbers.index(lastCheckpoint)
            modelName = "checkpoints/{}".format(checkpoints[checkpointInd])
            # Check if the latest checkpoint is the most recent line in the training CSV
            from pandas import read_csv
            # If this fails, you have to restart training from scratch anyways....
            CSVname = "training_log_{:02d}.csv".format(rank)
            trainingCSV = read_csv(CSVname)
            # If not, remove the extra lines
            trainingCSV_adj = trainingCSV[trainingCSV['epoch'] <= lastCheckpoint]
            trainingCSV_adj.to_csv(CSVname, index=False)
            initial_epoch = lastCheckpoint
            print("Resuming training, starting at epoch {}".format(lastCheckpoint))
            if initial_epoch == params['epochs']:
                print("Fully trained? initial_epoch = {} = {} = params['epochs']"
                        .format(initial_epoch, params['epochs']))
                exit()
        # Return the latest epoch and modelName
        return initial_epoch+1, modelName
    return None, None

if __name__ == "__main__":
    # Get mpi rank
    from getOneHot import getOneHot
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Load in the parameter files
    from json import load as loadf
    with open("params.json", 'r') as inFile:
        params = loadf(inFile)

    # Get data files and prep them for the generator
    from tensorflow import distribute as D
    callbacks = []
    devices = getDevices()
    print(devices)
    set_tf_config_mpi()
    strat = D.experimental.MultiWorkerMirroredStrategy()
    # Create network
    from sys import argv
    resume_training = False
    print(argv)
    if "resume_latest" in argv:
        resume_training = True

    with strat.scope():
        # Scheduler
        if isinstance(params["learning_rate"], str):
            # Get the string for the importable function
            lr = params["learning_rate"]
            from tensorflow.keras.callbacks import LearningRateScheduler
            # Use a dummy learning rate
            params["learning_rate"] = 0.1
            # model = create_model(**params)
            # Get the importable function
            lr = lr.split(".")
            baseImport = __import__(lr[0], globals(), locals(), [lr[1]], 0)
            lr = getattr(baseImport, lr[1])
            # Make a schedule
            lr = LearningRateScheduler(lr)
            callbacks.append(lr)
        # Resume Model?
        model_name = None
        if resume_training:
            initial_epoch, model_name = getInitialEpochsAndModelName(rank)
        if model_name is None:
            initial_epoch=0
            model = create_model(**params)
            resume_training = False
        else:
            from tensorflow.keras.models import load_model
            model = load_model(model_name)
    # Load data from disk
    import numpy
    if "root" in params.keys():
        root = params['root']
    else:
        root = "./"
    if "filename" in params.keys():
        filename = params["filename"]
    else:
        filename = "150MeV_all_shuffled_normed.csv"

    restricted = [
            'euc1', 'e1', 'x1', 'y1', 'z1',
            'euc2', 'e2', 'x2', 'y2', 'z2',
            'euc3', 'e3', 'x3', 'y3', 'z3',
            ]
    x, y = getOneHot("{}/{}".format(root, filename), restricted=restricted, **params)
    # val_filename = "150MeV_180kMUmin-stdCC_stitched_triples_dtot_trip_only.csv"
    # val_x, val_y = getOneHot("{}/{}".format(root, val_filename), restricted=restricted)
    val_x, val_y = None, None
    params["gbatch_size"] = params['batch_size'] * len(devices)
    print("x.shape =", x.shape)
    print("y.shape =", y.shape)
    print("epochs  =", params['epochs'], type(params['epochs']))
    print("batch   =", params['batch_size'], type(params['batch_size']))
    print("gbatch  =", params['gbatch_size'], type(params['gbatch_size']))
    # Load data into a distributed dataset
    # Dataset object does nothing in place:
    # https://stackoverflow.com/questions/55645953/shape-of-tensorflow-dataset-data-in-keras-tensorflow-2-0-is-wrong-after-conver
    from tensorflow.data import Dataset
    data = Dataset.from_tensor_slices((x, y))

    # Create validation set
    v = params['validation']
    if val_x is not None:
        vrecord = val_x.shape[0]
        val  = Dataset.from_tensor_slices((val_x, val_y))
        validation = val # data.take(vrecord)
    else:
        vrecord = int(x.shape[0]*v)
        validation = data.take(vrecord)
    validation = validation.batch(params['gbatch_size'])
    validation = validation.repeat(params['epochs'])
    # Validation -- need to do kfold one day
    # This set should NOT be distributed
    vsteps = vrecord // params['gbatch_size']
    if vrecord % params['gbatch_size'] != 0:
        vsteps += 1
    # Shuffle the data during preprocessing or suffer...
    # Parallel randomness == nightmare
    # data = data.shuffle(x.shape[0])
    # Ordering these two things is very important! 
    # Consider 3 elements, batch size 2 repeat 2
    # [1 2 3] -> [[1 2] [3]] -> [[1 2] [3] [1 2] [3]] (correct) batch -> repeat
    # [1 2 3] -> [1 2 3 1 2 3] -> [[1 2] [3 1] [2 3]] (incorrect) repeat -> batch
    # data = data.skip(vrecord)
    data    = data.batch(params['gbatch_size'])
    data    = data.repeat(params['epochs'])
    records = x.shape[0] # - vrecord
    steps   = records // params['gbatch_size']
    if records % params['gbatch_size']:
        steps += 1
    print("steps   =", steps)
    # Note that if we are resuming that the number of _remaining_ epochs has
    # changed!
    # The number of epochs * steps is the numbers of samples to drop
    print("initial   cardinality = ", data.cardinality())
    print("initial v cardinality = ", data.cardinality())
    data       = data.skip(initial_epoch*steps)
    validation = validation.skip(initial_epoch*vsteps)
    print("final     cardinality = ", data.cardinality())
    print("final v   cardinality = ", data.cardinality())
    # data = strat.experimental_distribute_dataset(data)
    # Split into validation and training
    callbacks  = createCallbacks(params, callbacks, rank, resume_training)
    print(callbacks)

    history = model.fit(data, epochs=params['epochs'],
            batch_size=params['gbatch_size'],
            steps_per_epoch=steps,
            verbose=0, 
            initial_epoch=initial_epoch,
            validation_data=validation,
            validation_steps=vsteps,
            callbacks=callbacks)
    if rank == 0:
        model.save("model-final")
    else:
        model.save("checkpoints/model-tmp")
