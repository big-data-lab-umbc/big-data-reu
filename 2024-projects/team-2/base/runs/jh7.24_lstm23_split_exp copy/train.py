import utils
from lightning import Trainer
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
import argparse
import yaml


PROJECT_BASE_PATH = '/nfs/rs/cybertrn/reu2024/team2/base/'


def get_args():
    parser = argparse.ArgumentParser(description='UMBC-HPCF PGML Team v0.0')
    parser.add_argument('-c', '--config', type=str, required=False, metavar="FILE", help='Path to config file')
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        cfg = yaml.safe_load(file)
    
    train_config = {'run_id': cfg['run_id'], 'pred_ckpt': cfg['pred_ckpt'], 'resume_ckpt': cfg['resume_ckpt'], 'mdl_key':cfg['mdl_key']}
    data_config = cfg['data']
    fit_config = cfg['fit']
    model_config = cfg['model']

    return train_config, data_config, fit_config, model_config, args.config


def main(train_config, data_config, fit_config, model_config):
    id = train_config['run_id']
    print("run id: ", id)

    # model and data
    model = utils.get_model(model_key=train_config['mdl_key'], params=model_config)
    data = utils.PGMLDataModule(train_data_path=data_config['train_data_path'], 
                          batch_size=data_config['batch_size'], 
                          val_split=data_config['val_split'],
                          split_func= 'sklearn'
                        )

    # loggers
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=PROJECT_BASE_PATH+'logs/tb_logs/'+str(id)+'/')
    csv_logger = pl_loggers.CSVLogger(save_dir=PROJECT_BASE_PATH+'logs/csv_logs/'+str(id)+'/')

    early_stopping_callback = EarlyStopping(monitor="valid_loss", min_delta=0.00, patience=fit_config['patience'], verbose=True, mode="min")
    # dumb checkpointing: saves every n epochs
    checkpoint_callback = ModelCheckpoint(save_top_k=1, every_n_epochs=fit_config['ckpt_freq'])
    trainer = Trainer(accelerator="gpu", devices=fit_config['n_devices'], num_nodes=fit_config['n_nodes'], strategy="ddp", 
                        logger=[tb_logger, csv_logger],
                        max_epochs=fit_config['max_epochs'],
                        callbacks=[early_stopping_callback, checkpoint_callback]
                        #default_root_dir='/nfs/rs/cybertrn/reu2024/team2/base/',  # dk if this is needed
                    )
    trainer.fit(model=model, datamodule=data, ckpt_path=(None if train_config['resume_ckpt'] == '' else train_config['resume_ckpt']))
    # for resuming: not sure how this deals with loading learning rate, etc, hparams not saved in this model, but old work? but lr?


if __name__ == "__main__":    
    # read in config
    train_config, data_config, fit_config, model_config, cfg = get_args()
    print('Using config: ', cfg)
    
    main(train_config, data_config, fit_config, model_config)
