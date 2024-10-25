import os
import gc
import copy
import shutil
import glob
import argparse
import torch
from pathlib import Path
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from deeptcr.utils import load_config
from deeptcr.trainer import LitNet
from deeptcr.datasets import AllTCRpepDataset


os.environ["WANDB_API_KEY"] = "" # input wandb api key for online log
torch.set_float32_matmul_precision('high')


def collect_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--run_name', type=str, default='debug')
    parser.add_argument('--mode', type=str, choices=['nested_CV'])
    parser.add_argument('--online', '-o', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume training from a previous interrupted session')    
    return parser
    

def main(args):
    config, config_name = load_config(args.config)
    L.seed_everything(config.train.seed)
    
    if args.resume_from:
        output_dir = Path(args.resume_from)
        print(f"Resuming training from: {output_dir}")    
        run_name = Path(output_dir).stem
    else:
        fn = time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())
        run_name = f'{args.run_name}_{args.mode}'
        output_dir = Path(config.logdir, run_name)
        output_dir.mkdir(parents=True, exist_ok=True)
    src_path = Path(args.config)
    dst_path = output_dir.joinpath(config_name+'.yaml')
    if src_path.resolve() != dst_path.resolve():
        shutil.copy(src_path, dst_path)
    else:
        print("Source and destination refer to the same file. Skipping copy.")    
    
    log_mode = 'online' if args.online else 'offline'
    logger = WandbLogger(
        project='TPNet', 
        name=run_name, 
        save_dir=config.logdir, 
        mode=log_mode, 
    )
    logger.log_hyperparams(config)
        
    if args.mode == 'nested_CV':
        total_data = pd.read_csv(config.bind_dataset.train_data)
        cv_id = np.asarray(list(total_data['partition']))
        cv_num = len(set(cv_id))
        _shape = total_data.shape[0]
        for cv_ in range(cv_num):
            cv_s = list(range(cv_num))
            cv_s.pop(cv_)
            for i, cv_2 in enumerate(cv_s):
                checkpoint_name = f't_{cv_}_v_{cv_2}'
                checkpoints_dir = Path(output_dir, 'checkpoints')
                checkpoint_pattern = f"{checkpoint_name}_epoch=*.ckpt"

                # Use glob to find all matching checkpoint files
                matching_checkpoints = glob.glob(str(checkpoints_dir / checkpoint_pattern))

                # Check if this combination has already been trained
                if matching_checkpoints:
                    print(f"Skipping already trained combination: {checkpoint_name}")
                    continue   
                             
                train_fold = copy.deepcopy(cv_s)
                train_fold.pop(i)
                trainset = AllTCRpepDataset(config.bind_dataset.train_data, config.bind_dataset, return_fold=train_fold)
                validset = AllTCRpepDataset(config.bind_dataset.train_data, config.bind_dataset, return_fold=[cv_2])
                train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.train.batch_size, shuffle=True, num_workers=0)
                valid_loader = torch.utils.data.DataLoader(validset, batch_size=config.train.batch_size, shuffle=False, num_workers=0)
                
                model = LitNet(checkpoint_name, config)
                
                checkpointer_cfg = config.train.get('checkpointer', {})
                checkpoint_callback = ModelCheckpoint(
                    dirpath=Path(output_dir, 'checkpoints'),
                    filename=checkpoint_name+"_{epoch:02d}",
                    verbose=True,
                    auto_insert_metric_name=True,
                    monitor=F"{checkpoint_name}/{config.train.checkpoint_monitor}",
                    mode=config.train.checkpoint_monitor_mode,
                    **checkpointer_cfg
                )
                
                earlystop_callback = EarlyStopping(
                    monitor=F"{checkpoint_name}/{config.train.checkpoint_monitor}",
                    mode=config.train.checkpoint_monitor_mode,
                    patience=config.train.earlystop_patience
                )
                
                lr_logger = LearningRateMonitor(logging_interval='epoch')
                
                trainer = L.Trainer(
                    max_epochs=config.train.epochs,
                    logger=logger,
                    callbacks=[checkpoint_callback, earlystop_callback, lr_logger],
                    strategy='auto',
                    accelerator='gpu',
                    devices=[args.gpu],
                    enable_progress_bar=True,
                )
                
                trainer.fit(model, train_loader, valid_loader)


if __name__ == '__main__':
    parser = collect_args()
    args = parser.parse_args()
    main(args)

            
            
            

    
    