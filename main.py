from model import create_model
from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import Callback as KCallback  
from datetime import datetime
from numpy import concatenate
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
    tf_config = {}
    hosts = comm.allgather(host)
    hosts = ["{}:8888".format(xhost) for xhost in hosts]
    tf_config['cluster'] = {'worker': hosts}
    tf_config['task']    = {'type': 'worker', 'index': rank}
    print(rank, tf_config)

    # Dump into the environment for tensorflow to use
    # This is the suggested method on the docs for some reason
    from os import environ
    import json
    environ['TF_CONFIG'] = json.dumps(tf_config)

def createCallbacks(params, callbacks, rank):
    callbacks.append(TimeHistory())
    csv = CSVLogger("training_log_{:02d}.csv".format(rank), append=False)
    callbacks.append(csv)
    checkpoints = ModelCheckpoint("checkpoints/model-"+"{epoch:03d}", 
            monitor='val_loss', period=64,verbose=0)
    callbacks.append(checkpoints)
    return callbacks

if __name__ == "__main__":
    # Get mpi rank
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    from json import load as loadf
    with open("params.json", 'r') as inFile:
        params = loadf(inFile)

    # Get data files and prep them for the generator
    from tensorflow import distribute as D
    callbacks = []
    devices = getDevices()
    set_tf_config_mpi()
    strat = D.experimental.MultiWorkerMirroredStrategy(
            communication=D.experimental.CollectiveCommunication.RING)
    # Create network
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
        model = create_model(**params)
    # Load data from disk
    import numpy
    root = ""
    filebase = ""
    x = numpy.load("{}/inputs.{}.npy".format(root, filebase))
    y = numpy.load("{}/outputs.{}.npy".format(root, filebase))
    print("x.shape =", x.shape)
    print("y.shape =", y.shape)
    print("epochs  =", params['epochs'], type(params['epochs']))
    print("batch   =", params['batch_size'], type(params['batch_size']))
    # Load data into a distributed dataset
    # Dataset object does nothing in place:
    # https://stackoverflow.com/questions/55645953/shape-of-tensorflow-dataset-data-in-keras-tensorflow-2-0-is-wrong-after-conver
    from tensorflow.data import Dataset
    data = Dataset.from_tensor_slices((x, y))
    v = 0.20
    vrecord = int(x.shape[0]*v)
    validation = data.take(vrecord)
    validation = validation.batch(params['batch_size'])
    validation = validation.repeat(params['epochs'])
    # Validation -- need to do kfold one day
    # This set should NOT be distributed
    data = data.skip(vrecord)
    vsteps = vrecord // params['batch_size']
    if vrecord % params['batch_size'] != 0:
        vsteps += 1
    # Shuffle the data during preprocessing or suffer...
    # Parallel randomness == nightmare
    # data = data.shuffle(x.shape[0])
    # Ordering these two things is very important! 
    # Consider 3 elements, batch size 2 repeat 2
    # [1 2 3] -> [[1 2] [3]] -> [[1 2] [3] [1 2] [3]] (correct) batch -> repeat
    # [1 2 3] -> [1 2 3 1 2 3] -> [[1 2] [3 1] [2 3]] (incorrect) repeat -> batch
    data = data.batch(params['batch_size'])
    data = data.repeat(params['epochs'])
    steps = x.shape[0] // params['batch_size']
    if x.shape[0] % params['batch_size']:
        steps += 1
    print("steps   =", steps)
    # data = strat.experimental_distribute_dataset(data)
    # Split into validation and training
    callbacks = createCallbacks(params, callbacks, rank)
    print(callbacks)
    history = model.fit(data, epochs=params['epochs'],
            batch_size=params['batch_size'],
            steps_per_epoch=steps,
            verbose=0, 
            validation_data=validation,
            validation_steps=vsteps,
            # validation_split=0.2, verbose=0, 
            callbacks=callbacks)
    # history = model.fit(inputs, outputs, batch_size=params['batch_size'],epochs=params['epochs'],
            # validation_split=0.2, verbose=0, 
            # callbacks=createCallbacks(params, callbacks, rank))
