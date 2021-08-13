# Copyright (c) 2020-present, Royal Bank of Canada.
# Copyright (c) 2018-present, Richard Shin
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#################################################################################################
# Original code is based on the seq2struct (https://arxiv.org/abs/1906.11790) implementation
# from https://github.com/rshin/seq2struct by Richard Shin
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#################################################################################################

import itertools
import operator

import attr
import numpy as np
import torch


def argsort(items, key=lambda x: x, reverse=False):
    orig_to_sort, sorted_items = zip(*sorted(
        enumerate(items), key=lambda x: key(x[1]), reverse=reverse))
    sort_to_orig = tuple(x[0] for x in sorted(
        enumerate(orig_to_sort), key=operator.itemgetter(1)))
    return sorted_items, sort_to_orig, orig_to_sort

def sort_lists_by_length(lists):
    return argsort(lists, key=len, reverse=True)

def batch_bounds_for_packing(lengths):
    last_length = 0
    count = len(lengths)
    result = []
    for i, (length, group) in enumerate(itertools.groupby(reversed(lengths))):
        if i > 0 and length <= last_length:
            raise ValueError('lengths must be decreasing and positive')
        result.extend([count] * (length - last_length))
        count -= sum(1 for _ in group)
        last_length = length
    return result

def _make_packed_sequence(data, batch_sizes):
    return torch.nn.utils.rnn.PackedSequence(data, torch.LongTensor(batch_sizes))


@attr.s(frozen=True)
class PackedSequencePlus:
    ps = attr.ib()
    lengths = attr.ib()
    sort_to_orig = attr.ib(converter=np.array)
    orig_to_sort = attr.ib(converter=np.array)

    @lengths.validator
    def descending(self, attribute, value):
        for x, y in zip(value, value[1:]):
            if not x >= y:
                raise ValueError('Lengths are not descending: {}'.format(value))

    def __attrs_post_init__(self):
        self.__dict__['cum_batch_sizes'] = np.cumsum([0] + self.ps.batch_sizes[:-1].tolist()).astype(np.int_)

    def apply(self, fn):
        return attr.evolve(self, ps=torch.nn.utils.rnn.PackedSequence(fn(self.ps.data), self.ps.batch_sizes))

    def with_new_ps(self, ps):
        return attr.evlove(self, ps=ps)

    def pad(self, batch_first, others_to_unsort=(), padding_value=0.0):
        padded, seq_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            self.ps, batch_first=batch_first, padding_value=padding_value)
        results = padded[self.sort_to_orig], [seq_lengths[i] for i in self.sort_to_orig]
        return results + tuple(t[self.sort_to_orig] for t in others_to_unsort)

    def cuda(self):
        if self.ps.data.is_cuda:
            return self
        return self.apply(lambda d: d.cuda())

    def raw_index(self, orig_batch_idx, seq_idx):
        result = np.take(self.cum_batch_sizes, seq_idx) + np.take(
            self.sort_to_orig, orig_batch_idx)
        if self.ps.data is not None:
            assert np.all(result < len(self.ps.data))
        return result

    def select(self, orig_batch_idx, seq_idx=None):
        if seq_idx is None:
            return self.ps.data[self.raw_index(orig_batch_idx, range(self.lengths[self.sort_to_orig[orig_batch_idx]]))]
        return self.ps.data[self.raw_index(orig_batch_idx, seq_idx)]

    def select_subseq(self, orig_batch_indices):
        lengths = [self.lengths[self.sort_to_orig[i]] for i in orig_batch_indices]
        return self.from_gather(
            lengths=lengths,
            map_index=self.raw_index,
            gather_from_indices=lambda indices: self.ps.data[torch.LongTensor(indices)])

    def orig_index(self, raw_idx):
        seq_idx = np.searchsorted(self.cum_batch_sizes, raw_idx, side='right') - 1
        batch_idx = raw_idx - self.cum_batch_sizes[seq_idx]
        orig_batch_idx = self.sort_to_orig[batch_idx]
        return orig_batch_idx, seq_idx

    def orig_batch_indices(self):
        result = []
        for bs in self.ps.batch_sizes:
            result.extend(self.orig_to_sort[:bs])
        return np.array(result)

    def orig_lengths(self):
        for sort_idx in self.sort_to_orig:
            yield self.lengths[sort_idx]

    def expand(self, k):
        v = self.ps.data
        ps_data = v.unsqueeze(1).repeat(1, k, *([1] * (v.dim() - 1))).view(-1, *v.shape[1:])
        batch_sizes = (np.array(self.ps.batch_sizes) * k).tolist()
        lengths = np.repeat(self.lengths, k).tolist()
        sort_to_orig = [exp_i for i in self.sort_to_orig for exp_i in range(i * k, i * k + k)]
        orig_to_sort = [exp_i for i in self.orig_to_sort for exp_i in range(i * k, i * k + k)]
        return PackedSequencePlus(_make_packed_sequence(ps_data, batch_sizes), lengths, sort_to_orig, orig_to_sort)

    @classmethod
    def from_lists(cls, lists, item_shape, tensor_type, item_to_tensor):
        result = tensor_type(sum(len(lst) for lst in lists), *item_shape)

        sorted_lists, sort_to_orig, orig_to_sort = sort_lists_by_length(lists)
        lengths = [len(lst) for lst in sorted_lists]
        batch_bounds = batch_bounds_for_packing(lengths)
        idx = 0
        for i, bound in enumerate(batch_bounds):
            for batch_idx, lst in enumerate(sorted_lists[:bound]):
                item_to_tensor(lst[i], batch_idx, result[idx])
                idx += 1

        result = torch.autograd.Variable(result)
        return cls(_make_packed_sequence(result, batch_bounds), lengths, sort_to_orig, orig_to_sort)

    @classmethod
    def from_gather(cls, lengths, map_index, gather_from_indices):
        sorted_lengths, sort_to_orig, orig_to_sort = argsort(lengths, reverse=True)
        batch_bounds = batch_bounds_for_packing(sorted_lengths)

        indices = []
        for seq_idx, bound in enumerate(batch_bounds):
            for batch_idx in orig_to_sort[:bound]:
                assert seq_idx < lengths[batch_idx]
                indices.append(map_index(batch_idx, seq_idx))
        result = gather_from_indices(indices)

        return cls(_make_packed_sequence(result, batch_bounds), sorted_lengths, sort_to_orig, orig_to_sort)

    @classmethod
    def cat_seqs(cls, items):
        batch_size = len(items[0].lengths)
        assert all(len(item.lengths) == batch_size for item in items[1:])

        unsorted_concat_lengths = np.zeros(batch_size, dtype=np.int)
        for item in items:
            unsorted_concat_lengths += list(item.orig_lengths())

        concat_data = torch.cat([item.ps.data for item in items], dim=0)
        concat_data_base_indices = np.cumsum([0] + [item.ps.data.shape[0] for item in items])

        item_map_per_batch_item = []
        for batch_idx in range(batch_size):
            item_map_per_batch_item.append([
                (item_idx, item, i)
                for item_idx, item in enumerate(items)
                for i in range(item.lengths[item.sort_to_orig[batch_idx]])])

        def map_index(batch_idx, seq_idx):
            item_idx, item, seq_idx_within_item = item_map_per_batch_item[batch_idx][seq_idx]
            return concat_data_base_indices[item_idx] + item.raw_index(batch_idx, seq_idx_within_item)

        return cls.from_gather(
            lengths=unsorted_concat_lengths,
            map_index=map_index,
            gather_from_indices=lambda indices: concat_data[torch.LongTensor(indices)])
