import torch
import lightning.pytorch as pl
from model import NN
from dataset import DM
from callbacks import MyPrintingCallback
from lightning.pytorch.callbacks import DeviceStatsMonitor, ModelSummary
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.strategies import DeepSpeedStrategy

# import config
from datetime import datetime
import time
from json import load as loadf
from getOneHot import getOneHot
import os
import sys

torch.set_float32_matmul_precision("medium")


if __name__ == "__main__":

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    gpu_ids = [int(id) for id in cuda_visible_devices.split(',')]

    # Load in the parameter files
    with open("params.json", 'r') as inFile:
        params = loadf(inFile)

    logger = CSVLogger(save_dir="logs/", name="my-model")
    strategy = "ddp"
    profiler = SimpleProfiler()

    model = NN(
        params = params
    )

    dm = DM(
        data_dir = params["root"],
        filename = params["filename"],
        batch_size = params["batch_size"], 
        num_workers = len(gpu_ids),
        params = params
    )

    trainer = pl.Trainer(
        default_root_dir = "./",
        strategy=strategy,
        profiler=profiler,
        logger=logger,
        accelerator='gpu',
        devices=gpu_ids,
        num_nodes=1,
        min_epochs=1,
        max_epochs=params["epochs"],
        precision=params["precision"],
        callbacks=[ModelSummary(max_depth=-1), MyPrintingCallback()],   
        enable_progress_bar=False     
    )

    ## training step
    start = time.time()
    trainer.fit(model, dm)
    end = time.time()
    elapsed = end - start
    print(f"Time elapsed {elapsed/60:.2f} min")


    with open("profiler_results.txt", "w") as file:
        file.write(profiler.summary())




    