""" Rex Scheduler for timm

rex scheduler, the code frame is imitated from step_lr.py in timm, 
and the code of the rex scheduler is from the paper 
REX: Revisiting Budgeted Training with an Improved Schedule 
https://arxiv.org/pdf/2107.04197.pdf

"""
import math
import torch

from timm.scheduler import Scheduler

import random

import logging
_logger = logging.getLogger(__name__)

class RexLRScheduler(Scheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_max: float = 0.01,
        lr_min: float = 0.,
        weight: float = 0.5,
        num_epochs=1,
        warmup_t=0,
        warmup_lr_init=0,
        t_in_epochs=True,
        noise_range_t=None,
        noise_pct=0.67,
        noise_std=1.0,
        noise_seed=42,
        initialize=True,
    ) -> None:
        super().__init__(
            optimizer,
            param_group_field="lr",
            noise_range_t=noise_range_t,
            noise_pct=noise_pct,
            noise_std=noise_std,
            noise_seed=noise_seed,
            initialize=initialize,
        )

        self.num_epochs = num_epochs
        self.lr_min = lr_min
        self.lr_max = lr_max
        if not self.lr_min <= self.lr_max:
            raise ValueError("lr_min must be <= lr_max")
        self.weight = weight
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs

        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        '''
        t:      current time(epoch) step
        T:      max time step
        rex:    lr = lr_0 * (1-t/T)/(0.5 + 0.5*(1-t/T))
        '''
        weight = self.weight
        current_ratio = float(t % self.num_epochs)
        left_ratio = float(self.num_epochs - t) / self.num_epochs
        val = self.lr_min + float(self.lr_max - self.lr_min) * (left_ratio / (1 - weight + weight * left_ratio))
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            lrs = [val for _ in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None
        
