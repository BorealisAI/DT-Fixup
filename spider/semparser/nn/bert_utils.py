# Copyright (c) 2020-present, Royal Bank of Canada.
# Copyright (c) 2019-present NAVER Corp.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#################################################################################################
# Original code is based on the sqlova (https://github.com/naver/sqlova/blob/master/sqlova/utils/utils_wikisql.py#L530) implementation
# from https://github.com/naver/sqlova by NAVER
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
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

BERT_PATH = "./bert"
BERT_LARGE_PATH = "./bert-large"
ROBERTA_PATH = "./roberta"
ROBERTA_LARGE_PATH = "./roberta-large"


def generate_inputs(tokenizer, raw_words, headers, tokenized=False, pretrained_encoder_type="bert"):
    tokens = []
    segment_ids = []

    if pretrained_encoder_type == "roberta":
        cls_token = "<s>"
        sep_token = "</s>"
    else:
        cls_token = "[CLS]"
        sep_token = "[SEP]"

    tokens.append(cls_token)
    segment_ids.append(0)

    # (start_idx, end_idx) tuple list of sentence word spans
    token_start_end_indx = []

    for i, raw_word in enumerate(raw_words):
        i_start = len(tokens)
        if tokenized:
            sub_tok = raw_word
        else:
            sub_tok = tokenizer.tokenize(raw_word)
        tokens += sub_tok
        i_end = len(tokens)
        token_start_end_indx.append((i_start, i_end))
        segment_ids += [0] * len(sub_tok)

    # segment_ids: [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1]
    # the last segment_id of [SEP] is 1, others are 0
    tokens.append(sep_token)
    segment_ids.append(0)

    # (start_idx, end_idx) tuple list of headers word spans
    header_start_end_indx = []
    name2i_start_end = {}
    for i, header in enumerate(headers):
        header_str = " ".join(header)
        if header_str in name2i_start_end:
            i_start_end = name2i_start_end[header_str]
        else:
            i_start = len(tokens)
            if tokenized:
                sub_tok = header
            else:
                sub_tok = tokenizer.tokenize(header)
            tokens += sub_tok
            i_end = len(tokens)
            i_start_end = (i_start, i_end)
            name2i_start_end[header_str] = i_start_end
            segment_ids += [1] * len(sub_tok)
            if i < len(headers) - 1:
                tokens.append(sep_token)
                segment_ids.append(0)
            elif i == len(headers) - 1:
                tokens.append(sep_token)
                segment_ids.append(1)
            else:
                raise EnvironmentError
        header_start_end_indx.append(i_start_end)
    return tokens, segment_ids, token_start_end_indx, header_start_end_indx


def get_wemb_bert(bert_config, model_bert, tokenizer, raw_words_batch, headers_batch,
                  num_out_layers_n=1, num_out_layers_h=1,
                  bert_no_grad=False, h_layer_from_top=True, tokenized=False, pretrained_encoder_type="bert"):
    # get contextual output of all tokens from bert
    all_encoder_layer, pooled_output, token_start_end_indx_batch, header_start_end_indx_batch, word_nums, length_flatted, header_nums = get_bert_output(
        model_bert, tokenizer, raw_words_batch, headers_batch, bert_no_grad=bert_no_grad, tokenized=tokenized, pretrained_encoder_type=pretrained_encoder_type)
    # all_encoder_layer: BERT outputs from all layers.
    # pooled_output: output of [CLS] vec.
    # token_start_end_indx_batch: start and end indices of token in tokens
    # header_start_end_indx: start and end indices of headers

    wemb_question = get_wemb_avg(token_start_end_indx_batch, word_nums,
                                 bert_config.hidden_size,
                                 bert_config.num_hidden_layers,
                                 all_encoder_layer,
                                 num_out_layers_n)
    wemb_header = get_wemb_header(header_start_end_indx_batch, length_flatted, sum(header_nums),
                                  bert_config.hidden_size,
                                  bert_config.num_hidden_layers, all_encoder_layer,
                                  num_out_layers_h, take_layer_from_top=h_layer_from_top)

    first_token_tensor = all_encoder_layer[-1][:, 0]
    return wemb_question, wemb_header, length_flatted, header_nums, first_token_tensor


def get_bert_output(model_bert, tokenizer, raw_words_batch, headers_batch, bert_no_grad=False, tokenized=False, pretrained_encoder_type="bert"):
    """
    INPUT
    :param model_bert: bert model
    :param tokenizer: WordPiece toknizerq
    :param raw_words_batch: Raw words in questions.
    :param headers_batch: Headers
    OUTPUT
    tokens_batch: BERT input tokens
    orig_to_tok_index: map the index of 1st-level-token to the index of 2nd-level-token
    tok_to_orig_index: inverse map.
    """

    word_nums = []
    header_nums = []  # The length of columns for each batch

    input_ids_batch = []
    tokens_batch = []
    segment_ids_batch = []
    input_mask_batch = []

    token_start_end_indx_batch = []  # index to retreive the position of contextual vector later.
    header_start_end_indx_batch = []

    # tokenize and get max_seq_length
    max_seq_length = 0

    input_ids_raw_batch = []
    segment_ids_raw_batch = []
    for b, raw_words in enumerate(raw_words_batch):
        # [CLS] nlu [SEP] col1 [SEP] col2 [SEP] ...col-n [SEP]
        # or [CLS] nlu [SEP] tab1 [SEP] tab2 [SEP] ...tab-n [SEP]
        # 1. Generate BERT inputs & indices.
        headers = headers_batch[b]
        tokens, segment_ids, token_start_end_indx, header_start_end_indx = generate_inputs(tokenizer, raw_words,
                                                                                           headers, tokenized, pretrained_encoder_type=pretrained_encoder_type)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        token_start_end_indx_batch.append(token_start_end_indx)
        header_start_end_indx_batch.append(header_start_end_indx)
        tokens_batch.append(tokens)
        header_nums.append(len(headers))
        word_nums.append(len(raw_words))

        max_seq_length = max(max_seq_length, len(input_ids))
        segment_ids_raw_batch.append(segment_ids)
        input_ids_raw_batch.append(input_ids)

    for input_ids, segment_ids in zip(input_ids_raw_batch, segment_ids_raw_batch):
        # Input masks
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        assert len(input_ids) == len(segment_ids)
        # 2. Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        input_ids_batch.append(input_ids)
        segment_ids_batch.append(segment_ids)
        input_mask_batch.append(input_mask)

    # Convert to tensor
    all_input_ids = torch.tensor(input_ids_batch, dtype=torch.long, device=device)
    all_input_mask = torch.tensor(input_mask_batch, dtype=torch.long, device=device)
    all_segment_ids = torch.tensor(segment_ids_batch, dtype=torch.long, device=device)

    # 3. Generate BERT output.
    if bert_no_grad:
        with torch.no_grad():
            last_encoder_layer, pooled_output, hidden_states = model_bert(
                all_input_ids, token_type_ids=all_segment_ids, attention_mask=all_input_mask)
    else:
        last_encoder_layer, pooled_output, hidden_states = model_bert(
            all_input_ids, token_type_ids=all_segment_ids, attention_mask=all_input_mask)

    embedding_output = hidden_states[0]
    all_encoder_layer = hidden_states[1:]

    # 4. generate length_flatted from header_start_end_indx_batch
    length_flatted = get_length_flatted(header_start_end_indx_batch)

    # length_flatted = torch.LongTensor(length_flatted)
    # token_start_end_indx_batch = torch.LongTensor(token_start_end_indx_batch)
    # header_start_end_indx = torch.LongTensor(header_start_end_indx)
    # word_nums = torch.LongTensor(word_nums)

    return all_encoder_layer, pooled_output, token_start_end_indx_batch, header_start_end_indx_batch, word_nums, length_flatted, header_nums


# @torch.jit.script
def get_wemb_avg(token_start_end_indx_batch: list, l_n: list, hS: int, num_hidden_layers: int,
                      all_encoder_layer: list, num_out_layers_n: int) -> torch.FloatTensor:
    """
    Get the representation of each words by averaging the token representations.
    l_n: word numbers
    hS: hidden size
    """
    bS = len(l_n)
    l_n_max = max(l_n)
    wemb_question = torch.zeros((int(bS), int(l_n_max), int(hS * num_out_layers_n)), device=device)
    for b, start_end_indx in enumerate(token_start_end_indx_batch):
        for i, start_end_i in enumerate(start_end_indx):
            assert len(start_end_i) == 2
            for i_layer_reverse in range(int(num_out_layers_n)):
                i_layer = num_hidden_layers - 1 - i_layer_reverse
                st = i_layer_reverse * hS
                ed = (i_layer_reverse + 1) * hS
                wemb_question[b, i, st:ed] = torch.mean(
                    all_encoder_layer[i_layer][b, start_end_i[0]:start_end_i[1], :], dim=0)

    return wemb_question

def get_wemb_header(header_start_end_indx_batch: list, length_flatted: list, num_of_all_headers_batch: int, hS: int,
               num_hidden_layers: int,
               all_encoder_layer: list, num_out_layers_h: int, take_layer_from_top=False) -> torch.FloatTensor:
    """
    As if
    [ [table-1-col-1-tok1, t1-c1-t2, ...],
       [t1-c2-t1, t1-c2-t2, ...].
       ...
       [t2-c1-t1, ...,]
    ]
    Get the representation of each tokens
    """
    length_flatted_max = max(length_flatted)
    wemb_header = torch.zeros((int(num_of_all_headers_batch), int(length_flatted_max),
                          int(hS * num_out_layers_h)), device=device)
    i_header = -1
    for b, header_start_end_indx in enumerate(header_start_end_indx_batch):
        for b1, header_start_end_i in enumerate(header_start_end_indx):
            assert len(header_start_end_i) == 2
            i_header += 1
            for i_layer_reverse in range(int(num_out_layers_h)):
                if take_layer_from_top:
                    i_layer = num_hidden_layers - 1 - i_layer_reverse
                else:
                    i_layer = i_layer_reverse

                st = i_layer_reverse * hS
                ed = (i_layer_reverse + 1) * hS
                wemb_header[i_header, 0:(header_start_end_i[1] - header_start_end_i[0]), st:ed] \
                    = all_encoder_layer[i_layer][b, header_start_end_i[0]:header_start_end_i[1], :]

    return wemb_header


def get_length_flatted(header_start_end_indx_batch):
    """
    # Treat columns as if it is a batch of natural language utterance with batch-size = # of columns * # of batch_size
    header_start_end_indx = [(17, 18), (19, 21), (22, 23), (24, 25), (26, 29), (30, 34)])
    """
    length_flatted = []
    for header_start_end_indx in header_start_end_indx_batch:
        for header_start_end_i in header_start_end_indx:
            length_flatted.append(header_start_end_i[1] - header_start_end_i[0])

    return length_flatted


def generate_perm_inv(perm):
    # Definitly correct.
    perm_inv = np.zeros(len(perm), dtype=np.int32)
    for i, p in enumerate(perm):
        perm_inv[int(p)] = i

    return perm_inv


def encode(lstm, wemb_l, l, return_hidden=False, hc0=None, take_feature='last'):
    """ [batch_size, max token length, dim_emb]
    """
    bS, mL, eS = wemb_l.shape

    # sort before packking
    l = np.array(l)
    perm_idx = np.argsort(-l)
    perm_idx_inv = generate_perm_inv(perm_idx)

    # pack sequence

    packed_wemb_l = nn.utils.rnn.pack_padded_sequence(wemb_l[perm_idx, :, :],
                                                      l[perm_idx],
                                                      batch_first=True)

    # Time to encode
    if hc0 is not None:
        hc0 = (hc0[0][:, perm_idx], hc0[1][:, perm_idx])

    packed_wemb_l = packed_wemb_l.float()  # I don't know why..
    packed_wenc, hc_out = lstm(packed_wemb_l, hc0)
    hout, cout = hc_out

    # unpack
    wenc, _l = nn.utils.rnn.pad_packed_sequence(packed_wenc, batch_first=True)

    if take_feature == 'last':
        # Take only final outputs for each columns.
        wenc = wenc[tuple(range(bS)), l[perm_idx] - 1]  # [batch_size, dim_emb]
        wenc.unsqueeze_(1)  # [batch_size, 1, dim_emb]
    elif take_feature == 'avg':
        lens = torch.from_numpy(l[perm_idx]).float().to(wenc.device).unsqueeze(1)
        wenc = wenc[tuple(range(bS)), :, :].sum(dim=1) / lens
        wenc.unsqueeze_(1)  # [batch_size, 1, dim_emb]
    elif take_feature == 'avg+last':
        last = wenc[tuple(range(bS)), l[perm_idx] - 1]
        lens = torch.from_numpy(l[perm_idx]).float().to(wenc.device).unsqueeze(1)
        wenc = last + wenc[tuple(range(bS)), :, :].sum(dim=1) / lens
        wenc.unsqueeze_(1)  # [batch_size, 1, dim_emb]
    elif take_feature == 'max+last':
        last = wenc[tuple(range(bS)), l[perm_idx] - 1]
        lens = torch.from_numpy(l[perm_idx]).float().to(wenc.device).unsqueeze(1)
        wenc = last + wenc[tuple(range(bS)), :, :].max(dim=1)[0]
        wenc.unsqueeze_(1)  # [batch_size, 1, dim_emb]
    elif take_feature == 'avg+max+last':
        last = wenc[tuple(range(bS)), l[perm_idx] - 1]
        lens = torch.from_numpy(l[perm_idx]).float().to(wenc.device).unsqueeze(1)
        wenc = (last + wenc[tuple(range(bS)), :, :].sum(dim=1) / lens
                + wenc[tuple(range(bS)), :, :].max(dim=1)[0])
        wenc.unsqueeze_(1)  # [batch_size, 1, dim_emb]

    wenc = wenc[perm_idx_inv]

    if return_hidden:
        hout = hout[:, perm_idx_inv].to(device)
        cout = cout[:, perm_idx_inv].to(device)

        return wenc, hout, cout
    else:
        return wenc


def encode_header(lstm, wemb_header, length_flatted, header_nums, take_feature='last'):
    wenc_header, hout, cout = encode(lstm, wemb_header, length_flatted, return_hidden=True, hc0=None, take_feature=take_feature)

    wenc_header = wenc_header.squeeze(1)
    hS = wenc_header.size(-1)

    wenc_hs = wenc_header.new_zeros(len(header_nums), max(header_nums), hS)
    wenc_hs = wenc_hs.to(device)

    # Re-pack according to batch.
    # ret = [B_NLq, max_len_headers_all, dim_lstm]
    st = 0
    for i, header_num in enumerate(header_nums):
        wenc_hs[i, :header_num] = wenc_header[st:(st + header_num)]
        st += header_num

    return wenc_hs

