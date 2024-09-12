"""
Utilization functions
"""

import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def str_to_bool(val):
    """Convert a string representation of truth to true (1) or false (0).
    Copied from the python implementation distutils.utils.strtobool

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    >>> str_to_bool('YES')
    1
    >>> str_to_bool('FALSE')
    0
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    if val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    raise ValueError('invalid truth value {}'.format(val))


def cosine_annealing(step, total_steps, lr_max, lr_min):
    """Cosine Annealing for learning rate decay scheduler"""
    return lr_min + (lr_max -
                     lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def keras_decay(step, decay=0.0001):
    """Learning rate decay in Keras-style"""
    return 1. / (1. + decay * step)


class SGDRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """SGD with restarts scheduler"""
    def __init__(self, optimizer, T0, T_mul, eta_min, last_epoch=-1):
        self.Ti = T0
        self.T_mul = T_mul
        self.eta_min = eta_min

        self.last_restart = 0

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        T_cur = self.last_epoch - self.last_restart
        if T_cur >= self.Ti:
            self.last_restart = self.last_epoch
            self.Ti = self.Ti * self.T_mul
            T_cur = 0

        return [
            self.eta_min + (base_lr - self.eta_min) *
            (1 + np.cos(np.pi * T_cur / self.Ti)) / 2
            for base_lr in self.base_lrs
        ]


def _get_optimizer(model_parameters, optim_config):
    """Defines optimizer according to the given config"""
    optimizer_name = optim_config['optimizer']

    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model_parameters,
                                    lr=optim_config['base_lr'],
                                    momentum=optim_config['momentum'],
                                    weight_decay=optim_config['weight_decay'],
                                    nesterov=optim_config['nesterov'])
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model_parameters,
                                     lr=optim_config['base_lr'],
                                     betas=optim_config['betas'],
                                     weight_decay=optim_config['weight_decay'],
                                     amsgrad=str_to_bool(
                                         optim_config['amsgrad']))
        
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(model_parameters,
                                    lr=optim_config['base_lr'],
                                    betas=optim_config['betas'],
                                    weight_decay=optim_config['weight_decay'],
                                    amsgrad=str_to_bool(
                                        optim_config['amsgrad']))
    else:
        print('Un-known optimizer', optimizer_name)
        sys.exit()

    return optimizer


def _get_scheduler(optimizer, optim_config):
    """
    Defines learning rate scheduler according to the given config
    """
    if optim_config['scheduler'] == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=optim_config['milestones'],
            gamma=optim_config['lr_decay'])

    elif optim_config['scheduler'] == 'sgdr':
        scheduler = SGDRScheduler(optimizer, optim_config['T0'],
                                  optim_config['Tmult'],
                                  optim_config['lr_min'])

    elif optim_config['scheduler'] == 'cosine':
        total_steps = optim_config['epochs'] * \
            optim_config['steps_per_epoch']

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                total_steps,
                1,  # since lr_lambda computes multiplicative factor
                optim_config['lr_min'] / optim_config['base_lr']))

    elif optim_config['scheduler'] == 'keras_decay':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: keras_decay(step))
    elif optim_config['scheduler'] == 'linear_decayLR':
        scheduler = LinearDecayLRScheduler(optimizer, warmup_updates=optim_config['decay_LR_warm'], max_updates=optim_config['decay_LR_max'])  
    elif optim_config['scheduler'] == 'LinearWarm_ExpDecay':
        scheduler = LinearWarm_ExpDecay(optimizer, warmup_epochs=optim_config['decay_LR_warm'], total_epochs=optim_config['decay_LR_max'], gamma=0.8)
    else:
        scheduler = None
    return scheduler


def create_optimizer(model_parameters, optim_config):
    """Defines an optimizer and a scheduler"""
    optimizer = _get_optimizer(model_parameters, optim_config)
    scheduler = _get_scheduler(optimizer, optim_config)
    return optimizer, scheduler


def seed_worker(worker_id):
    """
    Used in generating seed for the worker of torch.utils.data.Dataloader
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed, config = None):
    """ 
    set initial seed for reproduction
    """
    if config is None:
        raise ValueError("config should not be None")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = str_to_bool(config["cudnn_deterministic_toggle"])
        torch.backends.cudnn.benchmark = str_to_bool(config["cudnn_benchmark_toggle"])


class LinearWarm_ExpDecay(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, gamma=0.9, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.gamma = gamma
        super(LinearWarm_ExpDecay, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            lr = [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            # Exponential decay
            lr = [base_lr * (self.gamma ** (self.last_epoch - self.warmup_epochs)) for base_lr in self.base_lrs]
        return lr


class LinearDecayLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear learning rate scheduler with warm up."""

    def __init__(
        self,
        optimizer,
        warmup_updates: int,
        max_updates: int,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        self.warmup_updates = warmup_updates
        self.max_updates = max_updates
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    # return the leraning rate
    def get_lr(self):
        if self._step_count <= self.warmup_updates:
            return [self._step_count / self.warmup_updates * base_lr for base_lr in self.base_lrs]
        # elif self._step_count >= self.max_updates:
        #     return [0.0 for _ in self.base_lrs]
        else:
            pct_remaining = (self.max_updates - self._step_count) / (self.max_updates - self.warmup_updates)
            return [base_lr * pct_remaining for base_lr in self.base_lrs]


class TriStageLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear learning rate scheduler with warmup, hold, and decay."""

    def __init__(
        self,
        optimizer,
        warmup_updates: int,
        hold_updates: int,
        decay_updates: int,
        init_lr_scale: float = 0.01,
        final_lr_scale: float = 0.05,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        self.warmup_updates = warmup_updates
        self.hold_updates = hold_updates
        self.decay_updates = decay_updates
        self.init_lr_scale = init_lr_scale
        self.final_lr_scale = final_lr_scale

        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        if self._step_count <= self.warmup_updates:
            return [
                base_lr * (self.init_lr_scale + self._step_count / self.warmup_updates * (1 - self.init_lr_scale))
                for base_lr in self.base_lrs
            ]
        elif self.warmup_updates < self._step_count <= (self.warmup_updates + self.hold_updates):
            return list(self.base_lrs)
        elif self._step_count <= (self.warmup_updates + self.hold_updates + self.decay_updates):
            return [
                base_lr
                * math.exp(
                    math.log(self.final_lr_scale)
                    * (self._step_count - self.warmup_updates - self.hold_updates)
                    / self.decay_updates
                )
                for base_lr in self.base_lrs
            ]
        else:
            return [base_lr * self.final_lr_scale for base_lr in self.base_lrs]