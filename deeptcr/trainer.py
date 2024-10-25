import torch.optim as optim   
import numpy as np
from pathlib import Path
from sklearn.metrics import auc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import AUROC, Accuracy, Precision, Recall, F1Score, PrecisionRecallCurve
import lightning as L

from deeptcr.network import TECCNet, CNN_CDR123_global_max, TransformerPredictor_AB_cdr123_with_epi
from deeptcr.moe import MoE


import torchmetrics
from sklearn.metrics import roc_auc_score

_model_dict = {
    'tecc': TECCNet,
    'mixtcrpred': TransformerPredictor_AB_cdr123_with_epi,
    'nettcr': CNN_CDR123_global_max,
    'moe': MoE
}

class AUC01(torchmetrics.Metric):
    def __init__(self, compute_on_step: bool = False, dist_sync_on_step: bool = False):
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step)

        self.add_state("y_true", default=[], dist_reduce_fx="cat")
        self.add_state("y_pred", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds and target are expected to be torch.Tensors
        self.y_pred.append(preds.cpu().detach())
        self.y_true.append(target.cpu().detach())

    def compute(self):
        # Convert lists of tensors to single tensors
        y_pred = torch.cat(self.y_pred).numpy()
        y_true = torch.cat(self.y_true).numpy()

        # Check if y_true has more than one unique value
        if len(np.unique(y_true)) > 1:
            # Calculate AUC with a max_fpr of 0.1
            auc01 = roc_auc_score(y_true, y_pred, max_fpr=0.1)
            return auc01
        else:
            # Handle the case where only one class is present
            # You might want to log a warning or return a specific value
            print("Warning: Only one class present in y_true. AUC is not defined.")
            return torch.tensor(0.0)  # or return another appropriate value
        

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)
    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]
    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor        


class LitNet(L.LightningModule):
    def __init__(self, model_name, config) -> None:
        super().__init__() 
        self.config = config
        self.model = _model_dict[config.model_cls](**config.model_config)  
        self.model_name = model_name 
        
        ### Metrics
        self.auroc = AUROC(task='binary')
        self.accuracy = Accuracy(task='binary')
        self.precision = Precision(task='binary')
        self.recall = Recall(task='binary')
        self.f1 = F1Score(task='binary')
        self.pr_curve = PrecisionRecallCurve(task='binary')
        self.auc01 = AUC01()
        
        # loss
        self.loss = nn.BCELoss(reduction='none')
        self.lr_scheduler = None
        
        # Attributes to save batch ouputs
        self.train_step_outputs = []
        self.train_step_targets = []
        self.val_step_preds = []
        self.val_step_targets = []
        self.val_step_loss = []
        
        self.save_hyperparameters()
        
    def bind_step(self, batch):
        out = self.model(batch)
        pred = out['bind_pred']
        weight = batch['sample_weight']
        individual_losses = self.loss(pred, target:=batch['target'])

        # Apply sample weights and average
        weighted_loss = individual_losses * weight  # Element-wise multiplication
        celoss = weighted_loss.mean()  # Now reduce by taking the mean    
        
        # apply moe aux loss if existed
        if 'aux_loss' in out:
            aux_loss = out['aux_loss']   
        else:
            aux_loss = torch.zeros_like(celoss)
            
        loss = celoss + aux_loss
        losses = {
            'celoss': celoss.item(),
            'aux_loss': aux_loss.item()
        }
        return pred, target, loss, losses
            
    def training_step(self, batch, batch_idx):
        pred, target, loss, losses = self.bind_step(batch)
        self.log(F'{self.model_name}/train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(F'{self.model_name}/train_ce_loss', losses['celoss'], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(F'{self.model_name}/train_aux_loss', losses['aux_loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        pred, target, loss, losses = self.bind_step(batch)
        self.log(F'{self.model_name}/val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(F'{self.model_name}/val_ce_loss', losses['celoss'], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(F'{self.model_name}/val_aux_loss', losses['aux_loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        # save val batch output
        self.val_step_preds.append(pred.cpu())
        self.val_step_targets.append(target.cpu())   
        self.val_step_loss.append(loss.cpu().detach())       
        return loss
    
    def on_validation_epoch_end(self):
        pred = torch.hstack(self.val_step_preds)
        target = torch.hstack(self.val_step_targets)
    
        self.log(F'{self.model_name}/val_auroc', self.auroc(pred, target), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(F'{self.model_name}/val_auc01', self.auc01(pred, target), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(F'{self.model_name}/val_accuracy', self.accuracy(pred, target), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log(F'{self.model_name}/val_precision', self.precision(pred, target), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log(F'{self.model_name}/val_recall', self.recall(pred, target), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log(F'{self.model_name}/val_f1', self.f1(pred, target), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        precision, recall, thresholds = self.pr_curve(pred, target.type(torch.long))
        aupr = auc(recall.cpu().numpy(), precision.cpu().numpy())
        self.log(F'{self.model_name}/val_aupr', aupr, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        val_auroc = self.auroc(pred, target)
        avg_val_loss = torch.stack(self.val_step_loss).mean()
        auc_minus_loss = val_auroc - avg_val_loss
        self.log(f'{self.model_name}/val_auc_minus_loss', auc_minus_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        self.val_step_preds.clear()
        self.val_step_targets.clear()
        self.val_step_loss.clear()
        self.auroc.reset()
        self.auc01.reset()
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
    
    def predict_step(self, batch, batch_idx):
        out = self.model(batch)
        return out['bind_pred']
    
    def configure_optimizers(self):
        # Create the optimizer
        optimizer_cls = getattr(torch.optim, self.config.train.optimizer)
        optimizer = optimizer_cls(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            **self.config.train.optimizer_config
        )
        # For mixtcrpred cosine + warmup
        if self.config.train.scheduler == 'cosine_warmup':
            self.lr_scheduler = CosineWarmupScheduler(optimizer, **self.config.train.scheduler_config)
            return {"optimizer": optimizer}
        # Check if a scheduler is specified in the configuration
        elif self.config.train.scheduler:
            # Create the learning rate scheduler
            scheduler_cls = getattr(torch.optim.lr_scheduler, self.config.train.scheduler)
            scheduler_config = {
                "scheduler": scheduler_cls(optimizer, **self.config.train.scheduler_config)
            }
            
            # If a metric is specified for monitoring by the scheduler, add it to the config
            if self.config.train.scheduler_monitor:
                scheduler_config["monitor"] = f"{self.model_name}/{self.config.train.scheduler_monitor}"

            # Return the optimizer and scheduler configuration
            return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

        # Return only the optimizer if no scheduler is specified
        return {"optimizer": optimizer}
    
    def optimizer_step(self, *args, **kwargs):
        if self.lr_scheduler is not None:
            super().optimizer_step(*args, **kwargs)
            self.lr_scheduler.step()
        else:
            return super().optimizer_step(*args, **kwargs)


    
    