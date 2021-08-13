# Copyright (c) 2020-present, Royal Bank of Canada.
# Copyright (c) 2017-present, Pengcheng Yin
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#################################################################################################
# Code is based on the tranX (https://arxiv.org/abs/1810.02720) implementation
# from https://github.com/pcyin/tranX by Pengcheng Yin
#################################################################################################

import pickle
import numpy as np
import torch

from semparser.nn import nn_utils
from semparser.common.utils import cached_property
from semparser.common import registry

@registry.register('dataloader', 'base', 'from_bin_file')
class DataLoader(object):
    def __init__(self, examples):
        self.examples = examples

    @property
    def all_source(self):
        return [e.src_sent for e in self.examples]

    @property
    def all_targets(self):
        return [e.tgt_code for e in self.examples]

    @staticmethod
    def from_bin_file(file_path):
        if isinstance(file_path, list):
            file_paths = file_path
        else:
            file_paths = [file_path]
        examples = []
        for f in file_paths:
            with open(f, 'rb') as frdr:
                examples.extend(pickle.load(frdr))
        return DataLoader(examples)

    @staticmethod
    def from_multi_bin_files(file_paths):
        examples = []
        for f in file_paths:
            with open(f, 'rb') as frdr:
                examples.extend(pickle.load(frdr))

        return DataLoader(examples)

    def batch_iter(self, batch_size, shuffle=False):
        index_arr = np.arange(len(self.examples))
        if shuffle:
            np.random.shuffle(index_arr)

        batch_num = int(np.ceil(len(self.examples) / float(batch_size)))
        for batch_id in range(batch_num):
            batch_ids = index_arr[batch_size * batch_id: batch_size * (batch_id + 1)]
            batch_examples = [self.examples[i] for i in batch_ids]

            yield batch_examples

    def __len__(self):
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)


class Example(object):
    def __init__(self, src_sent, tgt_actions, tgt_code, tgt_ast, idx=0, meta=None):
        self.src_sent = src_sent
        self.tgt_code = tgt_code
        self.tgt_ast = tgt_ast
        self.tgt_actions = tgt_actions

        self.idx = idx
        self.meta = meta


class Batch(object):
    def __init__(self, examples, grammar, vocab, copy=True, cuda=False, tokenizer=None):
        self.examples = examples
        self.max_action_num = max(len(e.tgt_actions) for e in self.examples)

        self.src_sents = [e.src_sent for e in self.examples]
        self.src_sents_len = [len(e.src_sent) for e in self.examples]

        self.grammar = grammar
        self.vocab = vocab
        self.copy = copy
        self.cuda = cuda
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def get_frontier_prod_idx(self, t):
        ids = []
        for e in self.examples:
            if t < len(e.tgt_actions):
                ids.append(self.grammar.prod2id[e.tgt_actions[t].frontier_prod])
            else:
                ids.append(0)
        return torch.cuda.LongTensor(ids) if self.cuda else torch.LongTensor(ids)

    def get_frontier_type_idx(self, t):
        ids = []
        for e in self.examples:
            if t < len(e.tgt_actions):
                ids.append(self.grammar.type2id[e.tgt_actions[t].frontier_field.type])
            else:
                ids.append(0)
        return torch.cuda.LongTensor(ids) if self.cuda else torch.LongTensor(ids)

    @cached_property
    def src_sents_var(self):
        return nn_utils.to_input_tensor(self.src_sents, self.vocab.source, cuda=self.cuda)
