# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

def dot_prod_attention(h_t, src_encoding, src_encoding_att_linear, mask=None):
    att_weight = torch.bmm(src_encoding_att_linear, h_t.unsqueeze(2)).squeeze(2)
    if mask is not None:
        att_weight.data.masked_fill_(mask, -float('inf'))
    att_weight = F.softmax(att_weight, dim=-1)

    att_view = (att_weight.size(0), 1, att_weight.size(1))
    ctx_vec = torch.bmm(att_weight.view(*att_view), src_encoding).squeeze(1)

    return ctx_vec, att_weight

def length_array_to_mask_tensor(length_array, cuda=False, valid_entry_has_mask_one=False):
    max_len = max(length_array)
    batch_size = len(length_array)

    mask = np.zeros((batch_size, max_len), dtype=np.uint8)
    for i, seq_len in enumerate(length_array):
        if valid_entry_has_mask_one:
            mask[i][:seq_len] = 1
        else:
            mask[i][seq_len:] = 1

    mask = torch.BoolTensor(mask)
    return mask.cuda() if cuda else mask

def input_transpose(sents, pad_token):
    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    sents_t = []
    for i in range(max_len):
        sents_t.append([sents[k][i] if len(sents[k]) > i else pad_token for k in range(batch_size)])

    return sents_t

def word2id(sents, vocab):
    if type(sents[0]) == list:
        return [[vocab[w] for w in s] for s in sents]
    else:
        return [vocab[w] for w in sents]

def to_input_tensor(sequences, vocab, cuda=False):
    word_ids = word2id(sequences, vocab)
    sents_t = input_transpose(word_ids, vocab['<pad>'])

    sents_var = torch.LongTensor(sents_t)
    if cuda:
        sents_var = sents_var.cuda()

    return sents_var

def uniform_init(lower, upper, params):
    for p in params:
        p.data.uniform_(lower, upper)

def glorot_init(params):
    for p in params:
        if len(p.data.size()) > 1:
            init.xavier_normal_(p.data)

def t_fix(params, scale):
    for p in params:
        if len(p.data.size()) > 1:
            p.data.div_(scale)

def set_params(params, value):
    for p in params:
        p.data = torch.ones_like(p.data).normal_(std=1. / value)

def embedding_cosine(source, target, mask):
    embedding_differ = []
    for i in range(target.size(1)):
        emb = target[:, i:i + 1, :].expand_as(source)
        sim = F.cosine_similarity(emb, source, dim=-1)
        embedding_differ.append(sim)
    embedding_differ = torch.stack(embedding_differ, 1)
    embedding_differ.data.masked_fill_(mask.unsqueeze(2).expand(
        target.size(0), target.size(1), source.size(1)), 0)
    return embedding_differ

def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand
