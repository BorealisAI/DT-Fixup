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

import torch
from torch import nn
import torch.nn.functional as F

def clones(module_fn, N):
    return nn.ModuleList([module_fn() for _ in range(N)])

class Encoder(nn.Module):
    def __init__(self, layer, layer_size, N, tie_layers=False, use_lnorm=True):
        super(Encoder, self).__init__()
        if tie_layers:
            self.layer = layer()
            self.layers = [self.layer for _ in range(N)]
        else:
            self.layers = clones(layer, N)
        self.use_lnorm = use_lnorm
        self.norm = nn.LayerNorm(layer_size)

    def forward(self, x, relation, mask):
        for layer in self.layers:
            x = layer(x, relation, mask)
        if self.use_lnorm:
            x = self.norm(x)
        return x

class SelfAttentionWithRelations(nn.Module):
    def __init__(self, emb, heads=8, per_head=True):
        super(SelfAttentionWithRelations, self).__init__()

        assert emb % heads == 0, 'Embedding dim (%d) should be divisible by number of heads (%d)' % (emb, heads)

        self.emb = emb
        self.heads = heads
        self.per_head = per_head

        s = emb // heads

        if per_head:
            self.tokeys = nn.Linear(s, s, bias=False)
            self.toqueries = nn.Linear(s, s, bias=False)
            self.tovalues = nn.Linear(s, s, bias=False)
        else:
            self.tokeys = nn.Linear(emb, emb, bias=False)
            self.toqueries = nn.Linear(emb, emb, bias=False)
            self.tovalues = nn.Linear(emb, emb, bias=False)

        self.scale = 1 / (s ** 0.5)

        self.unifyheads = nn.Linear(heads * s, emb)

    def forward(self, x, relation_k, relation_v, mask=None):
        b, t, e = x.size()
        h = self.heads
        assert e == self.emb, 'Input embedding dim (%d) should match layer embedding dim (%d)' % (e, self.emb)

        s = e // h
        if self.per_head:
            x = x.view(b, t, h, s)
            e_k = relation_k.view(b, t, t, h, s)
            e_v = relation_v.view(b, t, t, h, s)
        keys = self.tokeys(x)
        queries = self.toqueries(x)
        values = self.tovalues(x)
        if not self.per_head:
            keys = keys.view(b, t, h, s)
            queries = queries.view(b, t, h, s)
            values = values.view(b, t, h, s)
            e_k = relation_k
            e_v = relation_v

        att_score_1 = torch.einsum('bind,bjnd->bijn', queries, keys)
        if self.per_head:
            att_score_2 = torch.einsum('biknd,bijnd->bijnk', queries.unsqueeze(2), e_k).squeeze(-1)
        else:
            att_score_2 = torch.einsum('bind,bijd->bijn', queries, e_k)
        att_score = (att_score_1 + att_score_2).mul_(self.scale)
        if mask is not None:
            mask = mask.unsqueeze(-1).expand_as(att_score)
            att_score = att_score.masked_fill_(mask == 0, -1e9)

        assert att_score.size() == (b, t, t, h)

        att_prob = F.softmax(att_score, 2)

        att_vec_1 = torch.einsum('bijn,bjnd->bind', att_prob, values)
        if self.per_head:
            att_vec_2 = torch.einsum('bijnk,bijnd->bindk', att_prob.unsqueeze(-1), e_v).squeeze(-1)
        else:
            att_vec_2 = torch.einsum('bijn,bijd->bind', att_prob, e_v)
        att_vec = att_vec_1 + att_vec_2
        att_vec = att_vec.contiguous().view(b, t, h * s)

        return self.unifyheads(att_vec)

class EncoderLayer(nn.Module):
    def __init__(self, emb, heads, ff_hidden_mult=4, dropout=0.1, num_relations=33, per_head=False, use_lnorm=True):
        super(EncoderLayer, self).__init__()

        self.attention = SelfAttentionWithRelations(emb, heads=heads, per_head=per_head)

        if per_head:
            self.relation_k_emb = nn.Embedding(num_relations, emb)
            self.relation_v_emb = nn.Embedding(num_relations, emb)
        else:
            self.relation_k_emb = nn.Embedding(num_relations, emb // heads)
            self.relation_v_emb = nn.Embedding(num_relations, emb // heads)

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.dropout = nn.Dropout(dropout)
        self.use_lnorm = use_lnorm

    def forward(self, x, relation, mask=None):
        relation_k = self.relation_k_emb(relation)
        relation_v = self.relation_v_emb(relation)

        attended = self.attention(x, relation_k, relation_v, mask)
        if self.use_lnorm:
            x = self.norm1(attended + x)
        else:
            x = attended + x
        x = self.dropout(x)
        fedforward = self.ff(x)
        if self.use_lnorm:
            x = self.norm2(fedforward + x)
        else:
            x = fedforward + x
        x = self.dropout(x)
        return x
