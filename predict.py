import os
import shutil
import argparse
import torch
from pathlib import Path
import numpy as np
import pickle
import lightning as L

from deeptcr.trainer import LitNet
from deeptcr.datasets import AllTCRpepDataset
from deeptcr.utils import load_config
from deeptcr.evaluation import output_res


torch.set_float32_matmul_precision('high')

def collect_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--ckpt_path', type=str, default=None)
    return parser

def main(args):
    config, config_name = load_config(args.config)
    L.seed_everything(config.train.seed)
    if args.ckpt_path is not None:
        ckpt_path = Path(args.ckpt_path)
        if ckpt_path.is_dir():
            ckpt_files = list(ckpt_path.glob('*.ckpt'))            
        else:
            ckpt_files = [ckpt_path]
    else:
        raise ValueError('Please specify the checkpoint path')        
    output_dir = ckpt_files[0].parent.parent
    
    testset = AllTCRpepDataset(
        config.bind_dataset.test_data,
        config.bind_dataset,
        cv_id=None,
        return_fold=None
    )
    test_loader = torch.utils.data.DataLoader(testset, batch_size=config.train.batch_size, shuffle=False, num_workers=4)
    
    bind_preds = []    
    for ckpt_file in ckpt_files:
        model = LitNet.load_from_checkpoint(ckpt_file, strict=False)
        model.eval()
        model.freeze()
        trainer = L.Trainer(
            strategy='auto',
            accelerator='gpu',
            devices=[0],
            logger=False,
            inference_mode=True
        )
        out = trainer.predict(model, test_loader)
        bind_preds.append(np.hstack(out))

    bind_pred_avg = np.mean(bind_preds, axis=0) 
    bind_targets = np.asarray([x['target'] for x in testset])
    epitope = np.asarray(testset.data.peptide)
    
    data_to_save = {
        "pred": bind_pred_avg,
        "target": bind_targets,
        "eptiope_names": epitope,
    }
    test_file_name = Path(config.bind_dataset.test_data).stem               
    with open(output_dir.joinpath(F'{test_file_name}_results.pickle'), 'wb') as f:
        pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
    output_res(bind_pred_avg, bind_targets, epitope, output_dir.joinpath(F'{test_file_name}_overall.txt'), mode='overall')
    output_res(bind_pred_avg, bind_targets, epitope, output_dir.joinpath(F'{test_file_name}_epitope.txt'), mode='epitope')
            

if __name__ == '__main__':
    parser = collect_args()
    args = parser.parse_args()
    main(args)

            
            
            

    
        