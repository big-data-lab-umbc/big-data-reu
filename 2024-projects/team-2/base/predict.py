import utils
import train
from train import get_args
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from lightning import Trainer
import os


def main(train_config, data_config, fit_config=None, model_config=None):
    # load model with saved weights
    model = utils.get_model(train_config['mdl_key'], pred=True).load_from_checkpoint(train_config["pred_ckpt"])  # **model_config <-- take this out asap, once the set hyperaamters goes into effect
                                                                                                                 # actually, this may be useful for out of date model checkpoints breaking
    # get test data
    print("FLAG INFO --> TESTING DATA PATH: ", data_config['test_data_path'])
    data = utils.PGMLDataModule(test_data_path=data_config['test_data_path'])

    # make predictions
    trainer = Trainer(accelerator="gpu", devices=1, num_nodes=1, strategy="ddp", default_root_dir=train.PROJECT_BASE_PATH+"logs/pred_logs/"+str(train_config['run_id'])+'/')
    y_pred = trainer.predict(model, dataloaders=data)
    y_pred = torch.reshape(torch.tensor(y_pred), data.y_truth.shape)  # a bit hacky, not sure why above line returns list
    print("sklearn acc: ", accuracy_score(y_pred, data.y_truth))
    
    # sanity check, of sorts
    print("trainer test check:")
    trainer = Trainer(accelerator="gpu", devices=1, num_nodes=1, strategy="ddp", enable_progress_bar=False, default_root_dir=train.PROJECT_BASE_PATH+"logs/pred_logs/"+str(train_config['run_id'])+'/test_check/')
    print(trainer.test(model, dataloaders=data))  # maybe replace with data.py class

    # save output
    dir = train.PROJECT_BASE_PATH+'eval/'+str(train_config['run_id'])
    print('WARNING: '+dir+' already exists') if os.path.isdir(dir) else os.mkdir(dir)  # make directory to save data inside
    np.save(dir+'/y_truth', data.y_truth.numpy())  # save the data inside the directory
    np.save(dir+'/y_pred', y_pred.numpy())
    np.savetxt(dir+'/y_truth.txt', data.y_truth.numpy())  # text files for readability
    np.savetxt(dir+'/y_pred.txt', y_pred.numpy())
    #np.save('/nfs/rs/cybertrn/reu2024/team2/base/eval/'+str(train_config['run_id'])+'/y_truth', data.y_truth.numpy())


if __name__ == "__main__":
    # read in config
    train_config, data_config, _, model_config, cfg = get_args()
    print('Using config: ', cfg)

    main(train_config, data_config, model_config=model_config)