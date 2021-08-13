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

import operator

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from semparser.model import batched_sequence
from semparser.model import transformer


def clamp(value, abs_max):
    value = max(-abs_max, value)
    value = min(abs_max, value)
    return value


def get_attn_mask(seq_lengths, device=None):
    max_length, batch_size = int(max(seq_lengths)), len(seq_lengths)
    attn_mask = torch.zeros(batch_size, max_length, max_length, dtype=torch.int64, device=device)
    for batch_idx, seq_length in enumerate(seq_lengths):
        attn_mask[batch_idx, :seq_length, :seq_length] = 1
    return attn_mask


def get_att_mask_2d(seq1, seq2, device=None):
    max_len1, bsz1 = int(max(seq1)), len(seq1)
    max_len2, bsz2 = int(max(seq2)), len(seq2)
    assert bsz1 == bsz2
    attn_mask = torch.zeros(bsz1, max_len1, max_len2, dtype=torch.int64, device=device)
    for batch_idx, (seq_len1, seq_len2) in enumerate(zip(seq1, seq2)):
        attn_mask[batch_idx, :seq_len1, :seq_len2].fill_(1)
    return attn_mask


class LookupEmbeddings(torch.nn.Module):
    def __init__(self, device, vocab, embedding, emb_size):
        super(LookupEmbeddings, self).__init__()
        self.device = device
        self.vocab = vocab
        self.emb_size = emb_size

        self.embedding = embedding

    def _compute_boundaries(self, token_lists):
        boundaries = [
            np.cumsum([0] + [len(token_list) for token_list in token_lists_for_item])
            for token_lists_for_item in token_lists
        ]

        return boundaries

    def _embed_token(self, token, batch_idx, out):
        emb = self.embedding.weight[self.vocab[token]]
        out.copy_(emb)

    def forward(self, token_lists):
        all_embs = batched_sequence.PackedSequencePlus.from_lists(
            lists=[
                [
                    token
                    for token_list in token_lists_for_item
                    for token in token_list
                ]
                for token_lists_for_item in token_lists
            ],
            item_shape=(self.emb_size,),
            tensor_type=torch.FloatTensor,
            item_to_tensor=self._embed_token)
        all_embs = all_embs.apply(lambda d: d.to(self.device))

        return all_embs, self._compute_boundaries(token_lists)


class LearnableEmbeddings(torch.nn.Module):
    def __init__(self, device, embedding, emb_size):
        super(LearnableEmbeddings, self).__init__()
        self.device = device
        self.emb_size = emb_size

        self.embedding = embedding

    def forward(self, token_lists):
        indices = batched_sequence.PackedSequencePlus.from_lists(
            lists=[
                [
                    token
                    for token_list in token_lists_for_item
                    for token in token_list
                ]
                for token_lists_for_item in token_lists
            ],
            item_shape=(1,),
            tensor_type=torch.LongTensor,
            item_to_tensor=lambda idx, batch_idx, out: out.fill_(idx))
        indices = indices.apply(lambda d: d.to(self.device))
        all_embs = indices.apply(lambda x: self.embedding(x.squeeze(-1)))

        return all_embs


class ConcatEmb(torch.nn.Module):
    def __init__(self):
        super(ConcatEmb, self).__init__()

    def forward(self, embs):
        return embs[0].apply(lambda _: torch.cat([item.ps.data for item in embs], -1))


class EmbLinear(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(EmbLinear, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, input_):
        all_embs = input_.apply(lambda d: self.linear(d))
        return all_embs


class BiLSTM(torch.nn.Module):
    def __init__(self, input_size, output_size, dropout, summarize):
        super(BiLSTM, self).__init__()

        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=output_size // 2,
            bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = output_size // 2
        self.summarize = summarize

    def forward(self, input_):
        all_embs, boundaries = input_

        desc_lengths = []
        batch_desc_to_flap_map = {}
        for batch_idx, boundaries_for_item in enumerate(boundaries):
            for desc_idx, (left, right) in enumerate(zip(boundaries_for_item, boundaries_for_item[1:])):
                desc_lengths.append((batch_idx, desc_idx, right - left))
                batch_desc_to_flap_map[batch_idx, desc_idx] = len(batch_desc_to_flap_map)

        remapped_ps_indices = []

        def rearranged_all_embs_map_index(desc_lengths_idx, seq_idx):
            batch_idx, desc_idx, _ = desc_lengths[desc_lengths_idx]
            return batch_idx, boundaries[batch_idx][desc_idx] + seq_idx

        def rearranged_all_embs_gather_from_indices(indices):
            batch_indices, seq_indices = zip(*indices)
            remapped_ps_indices[:] = all_embs.raw_index(batch_indices, seq_indices)
            return all_embs.ps.data[torch.LongTensor(remapped_ps_indices)]

        rearranged_all_embs = batched_sequence.PackedSequencePlus.from_gather(
            lengths=[length for _, _, length in desc_lengths],
            map_index=rearranged_all_embs_map_index,
            gather_from_indices=rearranged_all_embs_gather_from_indices)
        rev_remapped_ps_indices = tuple(x[0] for x in sorted(
            enumerate(remapped_ps_indices), key=operator.itemgetter(1)))

        if self.use_native:
            rearranged_all_embs = rearranged_all_embs.apply(self.dropout)
            output, (h, c) = self.lstm(rearranged_all_embs.ps)
        else:
            padded, seq_lengths = torch.nn.utils.rnn.pad_packed_sequence(rearranged_all_embs.ps)
            bsz = padded.size(1)
            output, _ = self.lstm(padded, [[
                (torch.zeros(bsz, self.hidden_size).cuda(),
                 torch.zeros(bsz, self.hidden_size).cuda()) for _ in range(2)]])
            h = torch.max(output, 0)[0]
        if self.summarize:
            if self.use_native:
                h = torch.cat((h[0], h[1]), dim=-1)

            new_all_embs = batched_sequence.PackedSequencePlus.from_gather(
                lengths=[len(boundaries_for_item) - 1 for boundaries_for_item in boundaries],
                map_index=lambda batch_idx, desc_idx: rearranged_all_embs.sort_to_orig[
                    batch_desc_to_flap_map[batch_idx, desc_idx]],
                gather_from_indices=lambda indices: h[torch.LongTensor(indices)])

            new_boundaries = [
                list(range(len(boundaries_for_item)))
                for boundaries_for_item in boundaries
            ]
        else:
            if not self.use_native:
                output = torch.nn.utils.rnn.pack_padded_sequence(output, seq_lengths)
            new_all_embs = all_embs.apply(
                lambda _: output.data[torch.LongTensor(rev_remapped_ps_indices)])
            new_boundaries = boundaries

        return new_all_embs, new_boundaries


class RelationalTransformer(nn.Module):
    def __init__(self, device, hidden_size, num_layers, num_heads=8, num_relations=33,
                 dropout=0.1, per_head=False, tie_layers=False, use_lnorm=True):
        super(RelationalTransformer, self).__init__()
        self.device = device
        self.encoder = transformer.Encoder(
            lambda: transformer.EncoderLayer(
                emb=hidden_size,
                heads=num_heads,
                dropout=dropout,
                num_relations=num_relations,
                per_head=per_head,
                use_lnorm=use_lnorm),
            hidden_size,
            num_layers,
            tie_layers,
            use_lnorm)

    def forward(self, relations, q_enc, c_enc, t_enc):
        enc = batched_sequence.PackedSequencePlus.cat_seqs((q_enc, c_enc, t_enc))
        enc_padded, _ = enc.pad(batch_first=True)
        q_enc_lengths = list(q_enc.orig_lengths())
        c_enc_lengths = list(c_enc.orig_lengths())
        t_enc_lengths = list(t_enc.orig_lengths())
        enc_lengths = list(enc.orig_lengths())
        enc_lengths2 = [q + c + t for q, c, t in zip(q_enc_lengths, c_enc_lengths, t_enc_lengths)]
        assert enc_lengths == enc_lengths2
        return self.forward_in(relations, enc_padded, q_enc_lengths, c_enc_lengths, t_enc_lengths, enc_lengths)

    def forward_in(self, relations, enc_padded, q_enc_lengths, c_enc_lengths, t_enc_lengths, enc_lengths):
        relations_t = torch.from_numpy(relations).to(self.device)

        mask = get_attn_mask(enc_lengths, device=self.device)
        enc_new = self.encoder(enc_padded, relations_t, mask=mask)

        def gather_from_enc_new(indices):
            batch_indices, seq_indices = zip(*indices)
            return enc_new[torch.LongTensor(batch_indices), torch.LongTensor(seq_indices)]

        q_enc_new = batched_sequence.PackedSequencePlus.from_gather(
            lengths=q_enc_lengths,
            map_index=lambda batch_idx, seq_idx: (batch_idx, seq_idx),
            gather_from_indices=gather_from_enc_new)
        c_enc_new = batched_sequence.PackedSequencePlus.from_gather(
            lengths=c_enc_lengths,
            map_index=lambda batch_idx, seq_idx: (batch_idx, q_enc_lengths[batch_idx] + seq_idx),
            gather_from_indices=gather_from_enc_new)
        t_enc_new = batched_sequence.PackedSequencePlus.from_gather(
            lengths=t_enc_lengths,
            map_index=lambda batch_idx, seq_idx: (
                batch_idx, q_enc_lengths[batch_idx] + c_enc_lengths[batch_idx] + seq_idx),
            gather_from_indices=gather_from_enc_new)
        return q_enc_new, c_enc_new, t_enc_new


class RelativeAttention(nn.Module):
    def __init__(self, q_size, k_size, emb, heads=8):
        super(RelativeAttention, self).__init__()

        self.e = emb
        self.h = heads
        self.s = emb // heads

        self.to_q = nn.Linear(q_size, emb, bias=False)
        self.to_k = nn.Linear(k_size, emb, bias=False)
        self.to_v = nn.Linear(k_size, emb, bias=False)

        self.scale = 1 / (self.s ** 0.5)

        self.unifyheads = nn.Linear(emb, emb)

    def forward(self, query, key, mask=None):
        b, q_t, q_e = query.size()
        b, k_t, k_e = key.size()

        q = self.to_q(query).view(b, q_t, self.h, self.s)
        k = self.to_k(key).view(b, k_t, self.h, self.s)
        v = self.to_v(key).view(b, k_t, self.h, self.s)

        att_score = torch.einsum('bind,bjnd->bijn', q, k)
        if mask is not None:
            mask = mask.unsqueeze(-1).expand_as(att_score)
            att_score = att_score.masked_fill_(mask == 0, -1e9)

        att_prob = F.softmax(att_score, 2)

        att_vec = torch.einsum('bijn,bjnd->bind', att_prob, v)
        att_vec = att_vec.contiguous().view(b, q_t, self.e)
        return self.unifyheads(att_vec)
