# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import time
from tqdm import tqdm
import random
import os

import torch
import numpy as np

from semparser.common import registry


@registry.register('trainer', 'base')
class ModelTrainer(object):
    def __init__(self, train_data, dev_data, evaluator, optimizer, exp_path='data/tmp/test',
                 max_epochs=100, valid_per_epoch=1, log_per_step=10, batch_size=16, small_bsz=16,
                 eval_after_epoch=0, anneal_max_epoch=0, random_seed=0, device='cpu'):
        self.train_data = train_data
        self.dev_data = dev_data
        self.evaluator = evaluator
        self.optimizer = optimizer
        self.exp_path = exp_path

        self.max_epochs = max_epochs
        self.valid_per_epoch = valid_per_epoch
        self.log_per_step = log_per_step
        self.batch_size = batch_size
        self.small_bsz = small_bsz
        self.eval_after_epoch = eval_after_epoch
        self.anneal_max_epoch = anneal_max_epoch
        self.random_seed = random_seed
        self.device = device

        self.logger = logging.getLogger()

        if anneal_max_epoch > 0:
            batch_num = int(np.ceil(len(self.train_data) / float(self.batch_size)))
            self.optimizer.anneal_max_step = anneal_max_epoch * batch_num
            msg = 'overriding anneal_max_step to {} based on anneal_max_epoch'
            self.logger.info(msg.format(self.optimizer.anneal_max_step))

    def setup_rseed(self):
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available() and self.device == 'cuda':
            torch.cuda.manual_seed_all(self.random_seed)

    def log_iter(self, train_iter, avg_loss):
        log_str = '[Iter %d] loss=%.5f lr=%.5f' % (train_iter, avg_loss, self.optimizer.cur_lr)
        self.logger.info(log_str)

    def estimate_transformer_input_stats(self, model):
        model.eval()
        src_max = schema_max = 0.
        for example in tqdm(self.train_data.examples):
            src_encodings, schema_encodings = model.encoder.score([example], return_trans_input=True)

            src_max = max(src_max, torch.norm(src_encodings, p=2, dim=-1).max().item())
            schema_max = max(schema_max, torch.norm(schema_encodings, p=2, dim=-1).max().item())

        return src_max, schema_max

    def get_gradients(self, model, batch):
        bsz = len(batch)
        start, end = 0, self.small_bsz

        report_loss = 0.0
        while start < bsz:
            small_batch = batch[start:end]
            small_bsz = len(small_batch)

            ret_val = model(small_batch)

            loss_val = ret_val * small_bsz / bsz
            report_loss += loss_val.detach().item() * bsz

            start = end
            end = start + self.small_bsz
            loss_val.backward()
            del loss_val, ret_val

        del batch
        return report_loss, bsz

    def run(self, model):
        self.setup_rseed()
        if torch.cuda.is_available() and self.device == 'cuda':
            model = model.cuda()
        self.optimizer.initialize(model)
        if self.optimizer.dt_fix:
            src_max, schema_max = self.estimate_transformer_input_stats(model)
            self.logger.info('src max: %s' % str(src_max))
            self.logger.info('schema max: %s' % str(schema_max))
            self.optimizer.dt_fixup(max(src_max, schema_max))

        epoch = train_iter = 0
        report_loss = report_examples = 0.
        history_dev_scores = []
        while epoch < self.max_epochs:
            epoch += 1

            model.train()
            tic = time.perf_counter()
            # --- training---
            self.logger.info('[Epoch %d] begin training' % epoch)
            for batch in self.train_data.batch_iter(self.batch_size, True):
                train_iter += 1
                self.optimizer.zero_grad()

                # back prop
                loss_val, bsz = self.get_gradients(model, batch)
                self.optimizer.step()

                # accumulate loss for report
                report_loss += loss_val
                report_examples += bsz
                if train_iter % self.log_per_step == 0:
                    self.log_iter(train_iter, report_loss / report_examples)
                    report_loss = report_examples = 0.

            toc = time.perf_counter()
            self.logger.info('[Epoch %d] training elapsed %.5f s' % (epoch, toc - tic))

            # --- evaluation ---
            is_better = False
            if self.valid_per_epoch <= 0:
                is_better = True
            elif (epoch % self.valid_per_epoch == 0 and epoch > self.eval_after_epoch):
                self.logger.info('[Epoch %d] begin validation' % epoch)
                tic = time.perf_counter()
                dev_score = self.evaluate(model, self.dev_data)

                toc = time.perf_counter()
                self.logger.info('[Epoch %d] dev acc: %.5f (took %ds)' % (
                    epoch, dev_score, toc - tic))
                self.logger.info('checkpoint: {}'.format(self.exp_path))

                is_better = history_dev_scores == [] or dev_score > max(history_dev_scores)
                history_dev_scores.append(dev_score)

            if is_better:
                model_path = os.path.join(self.exp_path, "model.bin")
                self.logger.info('save the current model...')
                model.save(model_path)

        if len(history_dev_scores) == 0:
            history_dev_scores.append(0)
        self.logger.info('best dev accuracy: %.5f' % max(history_dev_scores))
        self.logger.info('checkpoint: {}'.format(self.exp_path))

    def evaluate(self, model, test_data, return_decode_result=False, save_failed_samples=False):
        model.eval()

        batch_nums = int(np.ceil(len(test_data) / float(self.batch_size)))
        predictions = list()
        for batch in tqdm(test_data.batch_iter(self.batch_size), desc='Prediction', total=batch_nums):
            with torch.no_grad():
                preds = model.decode(batch)
            predictions.extend(preds)

        eval_result = self.evaluator(test_data.examples, predictions,
                                     self.exp_path if save_failed_samples else None)
        if return_decode_result:
            return eval_result, predictions
        else:
            return eval_result
