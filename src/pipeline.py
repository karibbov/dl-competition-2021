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


def run_pipeline(h='default', ssl='default'):
    # Pretrain with SSL
    if ssl == 'SSL':
        main(data_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset'),
             torch_model=USkipModel, num_epochs=1022, batch_size=64, learning_rate=2.223e-3,
             train_criterion=torch.nn.MSELoss, model_optimizer=torch.optim.Adam, scheduler_key='SGDR',
             data_augmentations=compose, save_model_str='models/', load_model_str=None, use_all_data_to_train=False,
             exp_name='SSL_Pretrain', config=None, training_type='SSL')

    if h == 'HPO':
        # minimum budget that BOHB uses
        min_budget = 3
        # largest budget BOHB will use
        max_budget = 510
        working_dir = os.curdir
        host = "localhost"
        port = 0
        run_id = 'bohb_run_1'
        n_bohb_iterations = 5#22
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
    else:
        # cos_hp = {'criterion': 'cross_entropy', 'data_augments': 'compose_nonorm', 'lr': 0.0015115146061245224,
        #           'optimizer': 'adam', 'scheduler': 'cosine_annealing', 'T': 1, 'eta_min': 3.8608850356656164e-05}
        cos3_hp = {'criterion': 'cross_entropy', 'data_augments': 'compose_nonorm', 'lr': 0.0017854146746553023,
                   'optimizer': 'adam', 'scheduler': 'cosine_annealing', 'T': 3, 'eta_min': 5.376674984956383e-06}
        # sgdr_hp = {'criterion': 'cross_entropy', 'data_augments': 'compose_nonorm', 'lr': 0.0011859223820560438,
        #            'optimizer': 'adam', 'scheduler': 'SGDR', 'T_mult': 1, 'cycle_len': 49}
        sgdr_n_hp = {'criterion': 'cross_entropy', 'data_augments': 'compose', 'lr': 0.001182754493778437,
                    'optimizer': 'adam', 'scheduler': 'SGDR', 'T_mult': 2, 'cycle_len': 1}
        sgdr_c_hp = {'criterion': 'cross_entropy', 'data_augments': 'compose_nonorm', 'lr': 0.001182754493778437,
                     'optimizer': 'adam', 'scheduler': 'SGDR', 'T_mult': 2, 'cycle_len': 1}

        val_acc = 0
        best_params = {}
        for h_params in [sgdr_c_hp, sgdr_n_hp, cos3_hp]:
            best_hypers = h_params
            print('Candidate:', best_hypers)
            result = main(data_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset'),
                         torch_model=TransferSkipModel, num_epochs=1022, batch_size=64, learning_rate=best_hypers['lr'],
                         train_criterion=loss_dict[best_hypers['criterion']], model_optimizer=opt_dict[best_hypers['optimizer']],
                         scheduler_key=best_hypers['scheduler'],
                         data_augmentations=eval(best_hypers['data_augments']), save_model_str='models/', load_model_str=get_saved_filename(),
                         use_all_data_to_train=False, exp_name='Final_HP_s', config=None, training_type='default')
            if val_acc < result[-1]:
                best_params = best_hypers
                val_acc = result[-1]

            print('BEST PARAMS: ', best_params)
            print('val_acc: ',result[-1], 'val_loss: ', result[-2], 'train_acc: ', result[-3], 'train_loss: ', result[0])

        best_hypers = best_params
    main(data_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset'),
          torch_model=TransferSkipModel, num_epochs=1022, batch_size=64, learning_rate=best_hypers['lr'],
          train_criterion=loss_dict[best_hypers['criterion']], model_optimizer=opt_dict[best_hypers['optimizer']],
          scheduler_key=best_hypers['scheduler'],
          data_augmentations=eval(best_hypers['data_augments']), save_model_str='models/', load_model_str=get_saved_filename(),
          use_all_data_to_train=True, exp_name='Final_Train', config=None, training_type='default')


if __name__ == '__main__':


    cmdline_parser = argparse.ArgumentParser('DL WS20/21 Competition')

    cmdline_parser.add_argument('-hpo', '--hpo_run',
                                default='default',
                                choices=['HPO', 'default'],
                                help="Run HPO or not: 'HPO' or 'default'")
    cmdline_parser.add_argument('-ssl', '--ssl_run',
                                default='default',
                                choices=['SSL', 'default'],
                                help="Run Self-Supervised Learning or not: 'SSL' or 'default'")


    args, unknowns = cmdline_parser.parse_known_args()
    log_lvl = logging.INFO
    logging.basicConfig(level=log_lvl)

    if unknowns:
        logging.warning('Found unknown arguments!')
        logging.warning(str(unknowns))
        logging.warning('These will be ignored')

    run_pipeline(h=args.hpo_run, ssl=args.ssl_run)



