# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

from semparser.data.rat import RatBatch
from semparser.common import registry
from semparser.nn.bert_utils import get_wemb_bert, encode_header
from semparser.nn.bert_utils import BERT_PATH, BERT_LARGE_PATH, ROBERTA_PATH, ROBERTA_LARGE_PATH

from semparser.model import batched_sequence
from semparser.model.rat_enc_modules import RelationalTransformer

def get_model(model_type):
    model = AutoModel.from_pretrained(model_type)
    model.encoder.output_hidden_states = True
    model.train()
    return model

@registry.register('encoder', 'rat-bert')
class RatBERTEncoder(nn.Module):
    def __init__(self, hidden_size, src_dropout=0.5, schema_dropout=0.5,
                 header_dropout=0.1, col_encodings_dropout=0.1, trans_dropout=0.0,
                 col_tbl_feat_agg='avg+last', freeze_bert=False, h_layer_from_bottom=False,
                 bert_out_n=1, bert_out_h=1, num_layers_rat=2, dt_fix=0, shuffle_tabs_cols=True,
                 pretrained_encoder_type='bert-base', device='cpu'):
        super(RatBERTEncoder, self).__init__()
        self.device = device
        self.src_dropout = nn.Dropout(src_dropout)
        self.schema_dropout = nn.Dropout(schema_dropout)
        self.header_dropout = nn.Dropout(header_dropout)
        self.col_encodings_dropout = nn.Dropout(col_encodings_dropout)
        self.hidden_size = hidden_size

        model_path = BERT_PATH
        if pretrained_encoder_type == 'bert-large':
            model_path = BERT_LARGE_PATH
        elif pretrained_encoder_type == 'roberta':
            model_path = ROBERTA_PATH
        elif pretrained_encoder_type == 'roberta-large':
            model_path = ROBERTA_LARGE_PATH

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.encoder_bert = get_model(model_path)

        if 'roberta' in pretrained_encoder_type:
            self.encoder_bert.config.type_vocab_size = 2
            self.encoder_bert.embeddings.token_type_embeddings = nn.Embedding(
                2, self.encoder_bert.config.hidden_size)
            self.encoder_bert.embeddings.token_type_embeddings.weight.data.normal_(
                mean=0.0, std=self.encoder_bert.config.initializer_range)

        self.encoder_bert.to(self.device)
        self.bert_config = self.encoder_bert.config
        self.bert_out_n = bert_out_n
        self.bert_out_h = bert_out_h
        self.bert_enc_size_n = self.bert_config.hidden_size * bert_out_n
        self.bert_enc_size_h = self.bert_config.hidden_size * bert_out_h
        self.pretrained_encoder_type = pretrained_encoder_type

        self.col_tbl_feat_agg = col_tbl_feat_agg
        self.freeze_bert = freeze_bert
        self.h_layer_from_bottom = h_layer_from_bottom
        self.shuffle_tabs_cols = shuffle_tabs_cols

        self.src_dimmap = nn.Linear(self.bert_enc_size_n, hidden_size, bias=False)

        self.schema_lstm = nn.LSTM(
            self.bert_enc_size_h, int(hidden_size / 2), bidirectional=True, batch_first=True)
        self.schema_dimmap = nn.Linear(hidden_size, hidden_size, bias=False)

        self.transformer = RelationalTransformer(
            self.device, hidden_size, num_layers_rat,
            dropout=trans_dropout, use_lnorm=not dt_fix)

        if self.device == 'cuda':
            self.new_tensor = torch.cuda.FloatTensor
        else:
            self.new_tensor = torch.FloatTensor

    def encoding_src_col(self, src_sents_word, tabs_cols_word, tokenized=False, pretrained_encoder_type="bert"):
        wemb_question, wemb_header, length_flatted, header_nums, first_token_tensor = get_wemb_bert(
            self.bert_config, self.encoder_bert, self.tokenizer, src_sents_word, tabs_cols_word,
            num_out_layers_n=self.bert_out_n,
            num_out_layers_h=self.bert_out_h,
            bert_no_grad=self.freeze_bert,
            h_layer_from_top=not self.h_layer_from_bottom,
            tokenized=tokenized,
            pretrained_encoder_type=pretrained_encoder_type)
        enc_layer = self.schema_lstm
        wemb_header = self.header_dropout(wemb_header)
        emb = encode_header(
            enc_layer, wemb_header, length_flatted=length_flatted, header_nums=header_nums,
            take_feature=self.col_tbl_feat_agg
        )
        return wemb_question, emb, first_token_tensor

    def example2batch(self, examples):
        return RatBatch(examples, None, None, cuda=self.device == 'cuda')

    def forward(self, batch, example_ids=None):
        self.schema_lstm.flatten_parameters()
        return self.score(batch, example_ids=example_ids)

    def score(self, examples, example_ids=None, return_encode_state=False, return_trans_input=False):
        if example_ids is not None:
            gathered = []
            for e in examples:
                if e.idx in example_ids.tolist():
                    gathered += [e]
            examples = gathered

        batch = self.example2batch(examples)

        tokenized = hasattr(batch, "bert_tokens")
        if tokenized:
            tabs_cols = batch.bert_tabs_cols
            sents = batch.bert_tokens
        else:
            tabs_cols = batch.origin_tabs_cols
            sents = batch.original_sents

        if self.shuffle_tabs_cols and self.training:
            old_pos_maps = []
            for i in range(len(batch.origin_tabs_cols)):
                shuffled_ids = np.array(list(range(0, len(tabs_cols[i]))))
                np.random.shuffle(shuffled_ids)
                old_pos_map = [-1] * len(shuffled_ids)
                for new_pos, old_pos in enumerate(shuffled_ids):
                    old_pos_map[old_pos] = new_pos
                tabs_cols[i] = np.array(tabs_cols[i])[shuffled_ids]
                old_pos_maps.append(old_pos_map)

        src_encodings, tab_col_encodings, first_token_tensor = self.encoding_src_col(
            sents, tabs_cols, tokenized, self.pretrained_encoder_type)

        if return_encode_state:
            return first_token_tensor
        if return_trans_input:
            return src_encodings, tab_col_encodings

        if self.shuffle_tabs_cols and self.training:
            for i in range(len(tabs_cols)):
                tab_col_encodings[i][:len(tabs_cols[i])] = tab_col_encodings[i][old_pos_maps[i]]

        src_encodings = self.src_dropout(src_encodings)
        src_encodings = self.src_dimmap(src_encodings)
        tab_col_encodings = self.schema_dropout(tab_col_encodings)
        tab_col_encodings = self.schema_dimmap(tab_col_encodings)

        question_lens = batch.src_sents_len
        column_nums = batch.col_len
        table_nums = batch.tab_len
        batch_size = len(batch.cols)
        cat = self.new_tensor(len(batch), max(batch.enc_len), self.hidden_size).zero_()
        for i in range(batch_size):
            cat[i][:batch.enc_len[i]] = torch.cat([
                src_encodings[i][:question_lens[i]],
                tab_col_encodings[i][table_nums[i]:table_nums[i] + column_nums[i]],
                tab_col_encodings[i][:table_nums[i]]], dim=0)
        enc_lengths = batch.enc_len
        q_enc, c_enc, t_enc = self.transformer.forward_in(
            batch.get_relation_matrix(), cat, question_lens, column_nums, table_nums, enc_lengths)

        enc = batched_sequence.PackedSequencePlus.cat_seqs((q_enc, c_enc, t_enc))
        enc_padded, _ = enc.pad(batch_first=True)
        enc_lengths = list(enc.orig_lengths())

        col_encodings, _ = c_enc.pad(batch_first=True)
        tab_encodings, _ = t_enc.pad(batch_first=True)

        col_tids = batch.get_column_ids_tensor()
        col_tab_encodings = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(
            tab_encodings, col_tids)])
        col_encodings = col_encodings + col_tab_encodings
        col_encodings = self.col_encodings_dropout(col_encodings)

        return batch, enc_padded, col_encodings, enc_lengths, column_nums
