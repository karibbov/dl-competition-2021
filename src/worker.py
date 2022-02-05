import logging
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import torch.nn as nn
import torch
from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB
from src.cnn import SkipModel, FunNet, TransferSkipModel
from src.main import *
# from src.pipeline import get_saved_filename
import os
import glob

def get_saved_filename() -> str:
    list_of_files = glob.glob('models/SSL*')
    latest_file = max(list_of_files, key=os.path.getctime)
    return os.path.basename(latest_file)

logging.getLogger('hpbandster').setLevel(logging.DEBUG)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Running on device: {device}")


class PyTorchWorker(Worker):
    def __init__(self, **kwargs):
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True
        # self.load_model_str = kwargs.get('load_model_str', None)
        # self.model =
        super().__init__(**kwargs)
        # self.train_loader = train_loader
        # self.validation_loader = validation_loader
        # self.test_loader = test_loader
        # self.input_shape = input_shape
        # self.num_classes = num_classes

    # @staticmethod
    # def get_model(config: CS.Configuration, input_shape, num_classes) -> nn.Module:
    #     return SkipModel(input_shape=input_shape, num_classes=num_classes)

    @staticmethod
    def get_configspace() -> CS.ConfigurationSpace:
        cs = CS.ConfigurationSpace()
        lr = CSH.UniformFloatHyperparameter('lr', lower=1e-6, upper=1e-1, log=True, default_value=1e-3)
        sgd_momentum = CSH.UniformFloatHyperparameter('sgd_momentum', lower=0.0, upper=0.99, default_value=0.90)
        optimizer = CSH.CategoricalHyperparameter('optimizer', choices=['adam', 'sgd'])
        criterion = CSH.CategoricalHyperparameter('criterion', choices=['cross_entropy'])
        scheduler = CSH.CategoricalHyperparameter('scheduler', choices=['SGDR', 'cosine_annealing',
                                                                        'cyclic', 'one_cycle'])
        data_augments = CSH.CategoricalHyperparameter('data_augments', choices=['compose', 'compose_nonorm'])

        cycle_len = CSH.UniformIntegerHyperparameter('cycle_len', lower=1, upper=50, default_value=1)
        T_mult = CSH.UniformIntegerHyperparameter('T_mult', lower=1, upper=2, default_value=2)
        T = CSH.UniformIntegerHyperparameter('T', lower=1, upper=20, default_value=4)
        eta_min = CSH.UniformFloatHyperparameter('eta_min', lower=1e-8, upper=1e-4, default_value=1e-8)
        m_lr = CSH.UniformIntegerHyperparameter('m_lr', lower=1, upper=4, default_value=1)
        step_size = CSH.UniformIntegerHyperparameter('step_size', lower=30, upper=2000, default_value=100)
        max_lr = CSH.UniformIntegerHyperparameter('max_lr', lower=1, upper=6, default_value=1)

        cs.add_hyperparameters([data_augments, lr, criterion, optimizer, sgd_momentum, scheduler,
                                cycle_len, T_mult, T, eta_min, max_lr, m_lr, step_size])

        momentum_cond = CS.EqualsCondition(sgd_momentum, optimizer, 'sgd')
        cs.add_condition(momentum_cond)

        cyc_l_cond = CS.EqualsCondition(cycle_len, scheduler, 'SGDR')
        T_mult_cond = CS.EqualsCondition(T_mult, scheduler, 'SGDR')
        T_cond = CS.EqualsCondition(T, scheduler, 'cosine_annealing')
        eta_min_cond = CS.EqualsCondition(eta_min, scheduler, 'cosine_annealing')
        m_lr_cond = CS.EqualsCondition(m_lr, scheduler, 'cyclic')
        step_s_cond = CS.EqualsCondition(step_size, scheduler, 'cyclic')
        max_lr_cond = CS.EqualsCondition(max_lr, scheduler, 'one_cycle')
        cs.add_conditions([cyc_l_cond, T_mult_cond, T_cond, eta_min_cond, m_lr_cond, step_s_cond, max_lr_cond])


        # cs.add_hyperparameter(lr)
        # nas_cs = CS.ConfigurationSpace()
        # ker1 = CSH.UniformIntegerHyperparameter('kernel1_size', lower=3, upper=21, q=2, default_value=15)
        # filters1 = CSH.UniformIntegerHyperparameter('filter1_n', lower=10, upper=50, q=10, default_value=20)
        # dilation1 = CSH.UniformIntegerHyperparameter('dilation1', lower=1, upper=2, default_value=1)
        # bn1 = CSH.CategoricalHyperparameter('bn1', choices=[True, False])
        # pool1 = CSH.CategoricalHyperparameter('pool1', choices=[True])
        #
        # n_layers = CSH.UniformIntegerHyperparameter('n_layers', lower=1, upper=2, default_value=1)
        #
        #
        # ker2 = CSH.UniformIntegerHyperparameter('kernel2_size', lower=3, upper=15, q=2, default_value=9)
        # filters2 = CSH.UniformIntegerHyperparameter('filter2_n', lower=10, upper=20, q=10, default_value=20)
        # dilation2 = CSH.UniformIntegerHyperparameter('dilation2', lower=1, upper=2, default_value=1)
        # bn2 = CSH.CategoricalHyperparameter('bn2', choices=[True, False])
        # pool2 = CSH.CategoricalHyperparameter('pool2', choices=[True])
        #
        # skip_layer = CSH.UniformIntegerHyperparameter('skip_layers', lower=1, upper=2, default_value=1)
        # # skip_layer = CSH.UniformIntegerHyperparameter('skip_layers', lower=1, upper=1, default_value=1)
        # fc1_nodes = CSH.UniformIntegerHyperparameter('fc1_nodes', lower=10, upper=30, q=2, default_value=16)

        # cs.add_hyperparameters([n_layers, ker1, filters1, dilation1, bn1, pool1,
        #                         ker2, filters2, dilation2, bn2, pool2, skip_layer, fc1_nodes])
        #
        # skip_cond = CS.EqualsCondition(skip_layer, n_layers, 2)
        # cs.add_condition(skip_cond)

        return cs

    def compute(self, config: CS.Configuration, budget: float, working_directory: str,
                *args, **kwargs) -> dict:
        """ Evaluate a function with the given config and budget and return a loss.
            Bohb tries to minimize the returned loss. In our case the function is
            the training and validation of a model, the budget is the number of
            epochs and the loss is the validation error.

        Args:
            config: Configuration space for joint HPO/NAS
            budget: number of epochs
            working_directory: not needed here !

        Returns:
            composition of loss, train, test & validation accuracy
            and PyTorch model converted to string.

        Note:
            Please notice that the optimizer is determined by the configuration space.
        """
        # data_augmentations = config['data_augments']
        loss_dict = {'cross_entropy': torch.nn.CrossEntropyLoss}  # Feel free to add more
        opt_dict = {'sgd': torch.optim.SGD, 'adam': torch.optim.Adam}  # Feel free to add more
        scheduler_dict = {'cosine_annealing': torch.optim.lr_scheduler.CosineAnnealingLR,
                          'SGDR': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts}
        result = main(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..', 'dataset'),
                      torch_model=TransferSkipModel,
                      num_epochs=int(budget),
                      batch_size=64,
                      learning_rate=config['lr'],
                      train_criterion=loss_dict[config['criterion']],
                      model_optimizer=opt_dict[config['optimizer']],
                      scheduler_key=config['scheduler'],
                      data_augmentations=eval(config['data_augments']),
                      load_model_str=get_saved_filename(),
                      save_model_str=None,
                      exp_name='workers',
                      use_all_data_to_train=False,
                      config=None,
                      other_params=config)
        # result = main(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..', 'dataset'),
        #               torch_model=SkipModel,
        #               num_epochs=int(budget),
        #               batch_size=64,
        #               learning_rate=config['lr'],
        #               train_criterion=loss_dict['cross_entropy'],
        #               model_optimizer=opt_dict['adam'],
        #               scheduler=scheduler_dict['SGDR'],
        #               data_augmentations=eval('compose'),
        #               save_model_str=None,
        #               load_model_str=None,
        #               exp_name='workers',
        #               use_all_data_to_train=False,
        #               config=None)


        return ({
            'loss': 1 - result[-1],  # remember: HpBandSter minimizes the loss!
            'info': {'test_accuracy': result[-1],
                     'train_accuracy': result[1],
                     'train_loss': result[0],
                     'model': str(SkipModel)}
        })