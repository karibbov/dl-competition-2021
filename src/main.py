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

def main(data_dir,
         torch_model,
         num_epochs=10,
         batch_size=64,
         learning_rate=0.001,
         train_criterion=torch.nn.CrossEntropyLoss,
         model_optimizer=torch.optim.Adam,
         scheduler_key=None,
         data_augmentations=None,
         save_model_str=None,
         load_model_str=None, #'SSL(128)_model_1643974116'
         use_all_data_to_train=False,
         exp_name='',
         config=None,
         training_type='default',
         other_params=None):
    """
    Training loop for configurableNet.
    :param torch_model: model that we are training
    :param data_dir: dataset path (str)
    :param num_epochs: (int)
    :param batch_size: (int)
    :param learning_rate: model optimizer learning rate (float)
    :param train_criterion: Which loss to use during training (torch.nn._Loss)
    :param model_optimizer: Which model optimizer to use during training (torch.optim.Optimizer)
    :param data_augmentations: List of data augmentations to apply such as rescaling.
        (list[transformations], transforms.Composition[list[transformations]], None)
        If none only ToTensor is used
    :param save_model_str: path of saved models (str)
    :param load_model_str: pretrained model, no pretrained model loaded if None
    :param use_all_data_to_train: indicator whether we use all the data for training (bool)
    :param exp_name: experiment name (str)
    :return:
    """

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print(data_augmentations)
    if data_augmentations is None:
        data_augmentations = transforms.ToTensor()
    elif isinstance(data_augmentations, list):
        data_augmentations = transforms.Compose(data_augmentations)
    elif not isinstance(data_augmentations, transforms.Compose):
        raise NotImplementedError
    SSL = training_type == 'SSL'
    # print(SSL)
    # Load the dataset
    train_data = ImageFolder(os.path.join(data_dir, 'train'), transform=data_augmentations)
    val_data = ImageFolder(os.path.join(data_dir, 'val'), transform=data_augmentations)
    test_data = ImageFolder(os.path.join(data_dir, 'test'), transform=data_augmentations)
    # random_loader = DataLoader(dataset=ConcatDataset([train_data]),
    #                                 batch_size=1,
    #                                 shuffle=True)
    # images = [transforms.ToPILImage()(next(iter(random_loader))[0]).convert('RGB') for i in range(17)]
    # orig, _ = train_data[-1]
    # print(orig)
    # plot_images(images)

    channels, img_height, img_width = train_data[0][0].shape

    # image size
    input_shape = (channels, img_height, img_width)

    # instantiate training criterion
    train_criterion = train_criterion().to(device)
    if SSL:
        train_criterion = torch.nn.MSELoss().to(device)
    score = []

    if use_all_data_to_train:
        train_loader = DataLoader(dataset=ConcatDataset([train_data, val_data, test_data]),
                                  batch_size=batch_size,
                                  shuffle=True)
        logging.warning('Training with all the data (train, val and test).')
    else:
        train_loader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True)
        # mean = 0
        # std = 0
        # c = 0
        # for image, label in train_loader:
        #     # print(image.size())
        #     mean += torch.mean(image, [0, 2, 3])
        #     # print(mean.size())
        #     std += torch.std(image, [0, 2, 3])
        #     c += 1
        # mean = mean / c
        # std = std / c
        # print("MEAN: ", mean)
        # print("STD: ", std)
        val_loader = DataLoader(dataset=val_data,
                                batch_size=batch_size,
                                shuffle=False)



    args_dict = {'input_shape': input_shape,
                 'num_classes': len(train_data.classes)}
    if config:
        args_dict.update({'block_config': config})
    if load_model_str:
        model_load_dir = os.path.join(os.getcwd(), 'models/')
        load_model_str = os.path.join(model_load_dir, load_model_str)
        args_dict.update({'load_model_str': load_model_str})

    model = torch_model(**args_dict).to(device)
    # instantiate optimizer

    optimizer = model_optimizer(model.parameters(), lr=learning_rate)
    if other_params and other_params['criterion'] == 'sgd':
        optimizer = model_optimizer(model.parameters(), lr=learning_rate, momentum=other_params['momentum'])
    # Info about the model being trained
    # You can find the number of learnable parameters in the model here
    logging.info('Model being trained:')
    summary(model, input_shape,
            device='cuda' if torch.cuda.is_available() else 'cpu')

    train_scores = []
    train_losses = []
    test_losses = []

    learning_rates = []
    t = 0
    cycles = 5
    cycle_len = 1
    T_mult = 2
    T = 5
    eta_min = 1e-8
    m_lr = 1
    step_size=100
    max_lr = 2
    if other_params:

        cycle_len = other_params.get('cycle_len', None)
        T_mult = other_params.get('T_mult', None)
        T = other_params.get('T', None)
        if T:
            T = num_epochs if int(num_epochs/T) < 1 else T
        eta_min = other_params.get('eta_min', None)
        if eta_min and eta_min > learning_rate:
            eta_min = learning_rate - learning_rate/100
        m_lr = other_params.get('m_lr', None)
        step_size = other_params.get('step_size', None)
        max_lr = other_params.get('max_lr', None)
    if type(optimizer).__name__ == 'Adam' and scheduler_key == 'cyclic':
        scheduler_key = 'one_cycle'
        max_lr = 100
    epoch_n = 120
    scheduler = None
    if scheduler_key == 'SGDR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, cycle_len, T_mult=T_mult)
    if scheduler_key == 'cosine_annealing':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(num_epochs/T), eta_min=eta_min)
    if scheduler_key == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=learning_rate, max_lr=learning_rate*(10^m_lr),
                                                  step_size_up=step_size, verbose=True)
    if scheduler_key == 'one_cycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate*(10^max_lr),
                                                        steps_per_epoch=batch_size, epochs=num_epochs)
    # print(type(scheduler).__name__)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.2, verbose=True, last_epoch=num_epochs*8)
    # Train the model
    for epoch in range(num_epochs):
        logging.info('#' * 50)
        logging.info('Epoch [{}/{}]'.format(epoch + 1, num_epochs))

        train_score, train_loss = train_fn(model, optimizer, train_criterion, train_loader,
                                           device, scheduler, epoch, learning_rates, SSL=SSL)
        logging.info('Train accuracy: %f', train_score)
        train_scores.append(train_score)
        train_losses.append(train_loss)
        # print(scheduler)
        # print(type(scheduler).__name__)
        if scheduler and type(scheduler).__name__ != "OneCycleLR" and type(scheduler).__name__ != 'CyclicLR':
            scheduler.step(epoch + t)
            learning_rates.append(scheduler.get_last_lr())
            print(f'Last Learning rate - {learning_rates[-1]}')

        # if (epoch + 1) % epoch_n == 0 and type(scheduler).__name__ == 'CosineAnnealingWarmRestarts':
        #     scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 60, T_mult=1)
        #     t += 1
        #     cycle_len = cycle_len - 1
        #     epoch_n += cycle_len
        if not use_all_data_to_train:
            # test_score = eval_fn(model, val_loader, device)
            test_score, test_loss = eval_loss(model, val_loader, train_criterion, device, SSL=SSL)
            logging.info('Validation accuracy: %f', test_score)
            score.append(test_score)
            test_losses.append(test_loss)

    if save_model_str:
        # Save the model checkpoint can be restored via "model = torch.load(save_model_str)"
        model_save_dir = os.path.join(os.getcwd(), save_model_str)

        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)
        model_id = str(int(time.time()))
        save_model_str = os.path.join(model_save_dir, exp_name + '_model_' + model_id)
        torch.save(model.state_dict(), save_model_str)
        plot_accuracy(train_scores, score, save=True, model=torch_model.__name__,
                      opt=model_optimizer.__name__, scheduler=type(scheduler).__name__,
                      lr=learning_rate, model_id=model_id)
        plot_loss(train_losses, test_losses, save=True, model=torch_model.__name__,
                  opt=model_optimizer.__name__, scheduler=type(scheduler).__name__,
                  lr=learning_rate, model_id=model_id)
        plot_lr(learning_rates, save=True, model=torch_model.__name__,
                  opt=model_optimizer.__name__, scheduler=type(scheduler).__name__,
                  lr=learning_rate, model_id=model_id)

    if not use_all_data_to_train:
        logging.info('Accuracy at each epoch: ' + str(score))
        logging.info('Mean of accuracies across all epochs: ' + str(100 * np.mean(score)) + '%')
        logging.info('Accuracy of model at final epoch: ' + str(100 * score[-1]) + '%')

        return [train_losses[-1], train_scores[-1], test_losses[-1], score[-1]]

if __name__ == '__main__':
    """
    This is just an example of a training pipeline.

    Feel free to add or remove more arguments, change default values or hardcode parameters to use.
    """
    loss_dict = {'cross_entropy': torch.nn.CrossEntropyLoss,
                 'nllloss': torch.nn.NLLLoss}  # Feel free to add more
    opti_dict = {'sgd': torch.optim.SGD, 'adam': torch.optim.Adam}  # Feel free to add more
    scheduler_dict = {'cosine_annealing': torch.optim.lr_scheduler.CosineAnnealingLR,
                      'SGDR': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts}

    cmdline_parser = argparse.ArgumentParser('DL WS20/21 Competition')

    cmdline_parser.add_argument('-m', '--model',
                                default='SampleModel',
                                help='Class name of model to train',
                                type=str)
    cmdline_parser.add_argument('-e', '--epochs',
                                default=50,
                                help='Number of epochs',
                                type=int)
    cmdline_parser.add_argument('-b', '--batch_size',
                                default=282,
                                help='Batch size',
                                type=int)
    cmdline_parser.add_argument('-D', '--data_dir',
                                default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     '..', 'dataset'),
                                help='Directory in which the data is stored (can be downloaded)')
    cmdline_parser.add_argument('-l', '--learning_rate',
                                default=2.244958736283895e-05,
                                help='Optimizer learning rate',
                                type=float)
    cmdline_parser.add_argument('-L', '--training_loss',
                                default='cross_entropy',
                                help='Which loss to use during training',
                                choices=list(loss_dict.keys()),
                                type=str)
    cmdline_parser.add_argument('-o', '--optimizer',
                                default='adam',
                                help='Which optimizer to use during training',
                                choices=list(opti_dict.keys()),
                                type=str)
    cmdline_parser.add_argument('-s', '--scheduler',
                                default='SGDR',
                                help='Which learning rate scheduler to use during training',
                                choices=list(scheduler_dict.keys()),
                                type=str)
    cmdline_parser.add_argument('-p', '--model_path',
                                default='models',
                                help='Path to store model',
                                type=str)
    cmdline_parser.add_argument('-v', '--verbose',
                                default='INFO',
                                choices=['INFO', 'DEBUG'],
                                help='verbosity')
    cmdline_parser.add_argument('-n', '--exp_name',
                                default='default',
                                help='Name of this experiment',
                                type=str)
    cmdline_parser.add_argument('-d', '--data-augmentation',
                                default='resize_and_colour_jitter',
                                help='Data augmentation to apply to data before passing to the model.'
                                     + 'Must be available in data_augmentations.py')
    cmdline_parser.add_argument('-a', '--use-all-data-to-train',
                                action='store_true',
                                help='Uses the train, validation, and test data to train the model if enabled.')
    cmdline_parser.add_argument('-t', '--training-type',
                                default='default',
                                choices=['SSL', 'default'],
                                help="Mode of training - Self-Supervised 'SSL' or 'default'")

    args, unknowns = cmdline_parser.parse_known_args()
    log_lvl = logging.INFO if args.verbose == 'INFO' else logging.DEBUG
    logging.basicConfig(level=log_lvl)

    if unknowns:
        logging.warning('Found unknown arguments!')
        logging.warning(str(unknowns))
        logging.warning('These will be ignored')

    main(
        data_dir=args.data_dir,
        torch_model=eval(args.model),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train_criterion=loss_dict[args.training_loss],
        model_optimizer=opti_dict[args.optimizer],
        scheduler=scheduler_dict[args.scheduler],
        data_augmentations=eval(args.data_augmentation),  # Check data_augmentations.py for sample augmentations
        save_model_str=args.model_path,
        exp_name=args.exp_name,
        use_all_data_to_train=args.use_all_data_to_train,
        training_type=args.training_type
    )
