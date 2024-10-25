import csv
import pickle
import numpy as np
from pathlib import Path
from sklearn.metrics import auc
from torchmetrics import AUROC, Accuracy, Precision, Recall, F1Score, PrecisionRecallCurve
from deeptcr.trainer import AUC01
import torch

def cal_metrics(pred, target):
    pred, target = torch.from_numpy(pred), torch.from_numpy(target)
    auroc = AUROC(task='binary')
    auc01 = AUC01()
    accuracy = Accuracy(task='binary')
    precision = Precision(task='binary')
    recall = Recall(task='binary')
    f1 = F1Score(task='binary')
    pr_curve = PrecisionRecallCurve(task='binary')
    
    auroc = auroc(pred, target)
    auc01 = torch.tensor(auc01(pred, target))
    accuracy = accuracy(pred, target)
    precision = precision(pred, target)
    recall = recall(pred, target)
    f1 = f1(pred, target)
    precision_, recall_, thresholds = pr_curve(pred, target.type(torch.int64))
    aupr = torch.tensor(auc(recall_, precision_))
    out = list(map(lambda x: np.around(x.numpy(), 4), [auroc, auc01, accuracy, precision, recall, f1, aupr]))
    return out

def output_res(pred, target, eptiope_names, output_path:Path, mode='overall'):
    """_summary_

    Args:
        pred (_type_): _description_
        target (_type_): _description_
        eptiope_names (_type_): _description_
        output_path (_type_): _description_
        mode (str, optional): [overall, epitope]. Defaults to 'overall'.
    """        
    output_path = Path(output_path).with_suffix('.csv')
    if mode == 'epitope':
        with open(output_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['epitope', 'total', 'pos', 'auroc', 'auc01', 'accuracy', 'precision', 'recall', 'f1', 'aupr'])
            outs_all = []
            eptiopes = sorted(list(set(eptiope_names)))
            for epitope in eptiopes:
                p_ = pred[eptiope_names == epitope]
                t_ = target[eptiope_names == epitope]
                total = len(t_)
                pos = np.sum(t_>0.0)
                outs = cal_metrics(p_, t_)
                outs_all.append(outs)
                writer.writerow([epitope, *([total, pos] + outs)])
            writer.writerow(['mean','','', *np.mean(outs_all, axis=0)])     
    elif mode == 'overall':
        total = len(target)
        pos = np.sum(target>0.0)
        auroc, auc01, accuracy, precision, recall, f1, aupr = cal_metrics(pred, target)
        with open(output_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['total', 'pos', 'auroc', 'auc01', 'accuracy', 'precision', 'recall', 'f1', 'aupr'])
            writer.writerow([total, pos, auroc, auc01, accuracy, precision, recall, f1, aupr])    
