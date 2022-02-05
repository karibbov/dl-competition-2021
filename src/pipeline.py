import os
import argparse
import logging
import time
import numpy as np
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchsummary import summary
from torchvision.datasets import ImageFolder

# from pytorch_metric_learning.losses import ContrastiveLoss
from src.cnn import *
from src.eval.evaluate import eval_fn, accuracy
from src.training import train_fn
from src.data_augmentations import *
from src.plotting import plot_accuracy, plot_loss, plot_lr
from src.utils import eval_loss, ContrastiveLoss

from src.worker import get_saved_filename
from src.main import main
from src.hpo import run_bohb
import glob
# import os


loss_dict = {'cross_entropy': torch.nn.CrossEntropyLoss}
opt_dict = {'sgd': torch.optim.SGD, 'adam': torch.optim.Adam}  # Feel free to add more
scheduler_dict = {'cosine_annealing': torch.optim.lr_scheduler.CosineAnnealingLR,
                  'SGDR': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts}


def run_pipeline():
    # Pretrain with SSL
    main(data_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset'),
         torch_model=USkipModel, num_epochs=1022, batch_size=64, learning_rate=2.223e-3,
         train_criterion=torch.nn.MSELoss, model_optimizer=torch.optim.Adam, scheduler_key='SGDR',
         data_augmentations=compose, save_model_str='models/', load_model_str=None, use_all_data_to_train=False,
         exp_name='SSL_Test', config=None, training_type='SSL')
    # minimum budget that BOHB uses
    min_budget = 3
    # largest budget BOHB will use
    max_budget = 511
    working_dir = os.curdir
    host = "localhost"
    port = 0
    run_id = 'bohb_run_1'
    n_bohb_iterations = 22
    res = run_bohb(
                host,
                port,
                run_id,
                n_bohb_iterations,
                working_dir,
                min_budget,
                max_budget,
                n_min_workers=4)
    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()


    best_hypers = id2config[incumbent]['config']
    print(f"BEST PARAMS: {best_hypers}")

    main(data_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset'),
         torch_model=TransferSkipModel, num_epochs=1022, batch_size=64, learning_rate=best_hypers['lr'],
         train_criterion=loss_dict[best_hypers['criterion']], model_optimizer=opt_dict[best_hypers['optimizer']],
         scheduler_key=best_hypers['scheduler'],
         data_augmentations=eval(best_hypers['data_augments']), save_model_str='models/', load_model_str=get_saved_filename(),
         use_all_data_to_train=True, exp_name='final_train', config=None, training_type='default')




if __name__ == '__main__':

    run_pipeline()



