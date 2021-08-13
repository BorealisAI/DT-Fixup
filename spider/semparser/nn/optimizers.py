# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging

import torch
from torch import optim

from semparser.common import registry
from semparser.nn import nn_utils


class OptimizerBase(object):

    def step(self):
        pass

    def initialize(self, model):
        pass

    def zero_grad(self):
        pass


@registry.register('optimizer', 'bot')
class BOTOptimizer(OptimizerBase):
    def __init__(self, lr=1e-3, eps=1e-8, clip_grad=5.0, warmup_step=-1, anneal_max_step=-1, anneal_power=0.5,
                 min_lr=1e-6, bert_lr_factor=0.008, num_layers_rat=8, dt_fix=True, glorot_init=False):
        self.lr = lr
        self.eps = eps
        self.optimizer = None
        self.clip_grad = clip_grad
        self.model = None
        self.cur_lr = lr

        self.warmup_step = warmup_step
        self.anneal_max_step = anneal_max_step
        self.anneal_power = anneal_power
        self.min_lr = min_lr
        self.bert_lr_factor = bert_lr_factor

        self.num_layers_rat = num_layers_rat
        self.dt_fix = dt_fix
        self.glorot_init = glorot_init

        self.iters = 0

    def is_tfix_params(self, name):
        if "transformer.encoder.layers" in name:
            if "tovalues" in name:
                return True
            if "relation_v_emb" in name:
                return True
            if "unifyheads" in name:
                return True
            if "ff" in name:
                return True
        return False

    def dt_fixup(self, max_norm):
        dtfix_params = [p for n, p in self.model.named_parameters() if self.is_tfix_params(n)]
        norm_factor = 4 * (max_norm ** 2) + 2 * max_norm + 2
        factor = (norm_factor * self.num_layers_rat) ** 0.5
        nn_utils.t_fix(dtfix_params, factor)

        logger = logging.getLogger()
        logger.info('DT-Fixup norm factor: %d' % norm_factor)
        logger.info('DT-Fixup done!')

    def initialize(self, model):
        logger = logging.getLogger()
        self.model = model

        bert_params = [
            p for n, p in model.named_parameters() if 'encoder_bert' in n
        ]
        other_params = [
            p for n, p in model.named_parameters() if 'encoder_bert' not in n
        ]
        trainable_parameters = bert_params + other_params
        optimizer_grouped_parameters = [
            {'params': other_params,
             'lr': self.lr, 'eps': self.eps},
            {'params': bert_params,
             'lr': self.lr * self.bert_lr_factor, 'eps': self.eps}]

        self.optimizer = optim.Adam(optimizer_grouped_parameters)

        total_params = sum(p.numel() for p in model.parameters())
        total_params_trainable = sum(p.numel() for p in trainable_parameters)

        logger.info("total_params_trainable: {}/{}".format(total_params_trainable, total_params))

        # initialization
        if self.glorot_init:
            nn_utils.glorot_init(other_params)
            logger.info('glorot_init done')

    def step(self):
        if self.optimizer is None:
            raise ValueError("Initialize first!")

        self.iters += 1

        if self.clip_grad > 0.:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)

        if self.iters <= self.warmup_step:
            self.set_lr(self.get_warmup_lr())

        elif self.iters > self.warmup_step:
            self.set_lr(self.get_anneal_lr())

        self.optimizer.step()

    def zero_grad(self):
        if self.optimizer is None:
            raise ValueError("Initialize first!")
        self.optimizer.zero_grad()

    def get_warmup_lr(self):
        return self.lr * min(1., self.iters / self.warmup_step)

    def get_anneal_lr(self):
        factor = (self.iters - self.warmup_step) / \
            (self.anneal_max_step - self.warmup_step)
        factor = min(1., factor)

        lr = self.lr * ((1 - factor) ** self.anneal_power)

        if self.min_lr > 0:
            lr = max(self.min_lr, lr)

        return lr

    def set_lr(self, lr):
        self.cur_lr = lr
        params = self.optimizer.param_groups

        params[0]['lr'] = lr
        params[1]['lr'] = lr * self.bert_lr_factor
