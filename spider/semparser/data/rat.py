# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from itertools import chain
from collections import defaultdict

import numpy as np
import torch

from semparser.data.base import Example, Batch
from semparser.common.utils import cached_property
from semparser.nn import nn_utils


class RatExample(Example):
    def __init__(self, question, original, tab_names, col_names, origin_tab_names, origin_col_names, tab_types, col_types,
                 sql, schema, tgt_actions, tgt_code, tgt_ast, idx=0, meta=None, col_valus=None, uri=None, db_id=None):
        super(RatExample, self).__init__(question, tgt_actions, tgt_code, tgt_ast, idx, meta)
        self.original = original
        self.tab_names = tab_names
        self.col_names = col_names
        self.origin_tab_names = origin_tab_names
        self.origin_col_names = origin_col_names
        self.tab_types = tab_types
        self.col_types = col_types
        self.sql = sql
        self.schema = schema
        self.uri = uri

        self.column_names = [x for x in schema._table['exp_column_names']]
        self.table_names = schema._table['exp_table_names']
        self.column_names[0] = (len(self.table_names) - 1, self.column_names[0][1])
        self.foreign_keys = schema._table['exp_foreign_keys']
        self.primary_keys = schema._table['exp_primary_keys']

        self.col2col = self.column_column_type()
        self.col2tab = self.column_table_type()
        self.col2tok = self.column_text_type()

        self.tab2col = self.table_column_type()
        self.tab2tab = self.table_table_type()
        self.tab2tok = self.table_text_type()

        self.tok2col = self.text_column_type()
        self.tok2tab = self.text_table_type()
        self.tok2tok = self.relative_position()

        self.col_value = col_valus
        self.db_id = db_id

    def table_id(self, col_id):
        return self.column_names[col_id][0]

    def is_primary(self, col_id):
        return col_id in self.primary_keys

    def column_column_type(self):
        '''
        0: Column Identity
        1: Same Table
        2: Foreign Key Column Foward
        3: Foreign Key Column Backward
        4: Column Column
        '''
        col_num = len(self.column_names)
        type_matrix = []
        for i in range(col_num):
            type_vector = []
            for j in range(col_num):
                if i == j:
                    type_vector.append(0)
                elif self.table_id(i) == self.table_id(j):
                    type_vector.append(1)
                elif [i, j] in self.foreign_keys or [j, i] in self.foreign_keys:
                    if self.is_primary(i):
                        type_vector.append(2)
                    else:
                        type_vector.append(3)
                else:
                    type_vector.append(4)
            type_matrix.append(type_vector)
        return type_matrix

    def table_table_type(self):
        '''
        5: Table Identity
        6: Foreign Key Table Forward
        7: Foreign Key Table Backward
        8: Foreign Key Table Bidirection
        9: Table Table
        '''
        edges = defaultdict(list)
        for x, y in self.foreign_keys:
            x_tab = self.table_id(x)
            y_tab = self.table_id(y)
            if self.is_primary(x):
                edges[x_tab].append(y_tab)
            if self.is_primary(y):
                edges[y_tab].append(x_tab)

        tab_num = len(self.table_names)
        type_matrix = []
        for i in range(tab_num):
            type_vector = []
            for j in range(tab_num):
                if i == j:
                    type_vector.append(5)
                elif (i in edges[j]) and (j in edges[i]):
                    type_vector.append(8)
                elif i in edges[j]:
                    type_vector.append(6)
                elif j in edges[i]:
                    type_vector.append(7)
                else:
                    type_vector.append(9)
            type_matrix.append(type_vector)
        return type_matrix

    def column_table_type(self):
        '''
        10: Primary Key Forward
        11: Belongs To Forward
        12: Column Table
        '''
        col_num = len(self.column_names)
        tab_num = len(self.table_names)
        type_matrix = []
        for i in range(col_num):
            type_vector = []
            for j in range(tab_num):
                if self.table_id(i) == j:
                    if self.is_primary(i):
                        type_vector.append(10)
                    else:
                        type_vector.append(11)
                else:
                    type_vector.append(12)
            type_matrix.append(type_vector)
        return type_matrix

    def table_column_type(self):
        '''
        13: Primary Key Reverse
        14: Belongs To Reverse
        15: Table Column
        '''
        col_num = len(self.column_names)
        tab_num = len(self.table_names)
        type_matrix = []
        for i in range(tab_num):
            type_vector = []
            for j in range(col_num):
                if self.table_id(j) == i:
                    if self.is_primary(j):
                        type_vector.append(13)
                    else:
                        type_vector.append(14)
                else:
                    type_vector.append(15)
            type_matrix.append(type_vector)
        return type_matrix

    def relative_position(self):
        '''
        16: -2
        17: -1
        18: 0
        19: +1
        20: +2
        '''
        sent_len = len(self.src_sent)
        pos_mat = []
        for i in range(sent_len):
            pos_vec = []
            for j in range(sent_len):
                if j - i < -1:
                    pos_vec.append(16)
                elif j - i == -1:
                    pos_vec.append(17)
                elif j - i == 0:
                    pos_vec.append(18)
                elif j - i == 1:
                    pos_vec.append(19)
                else:
                    pos_vec.append(20)
            pos_mat.append(pos_vec)
        return pos_mat

    def text_table_type(self):
        '''
        21: Text Table Exact Match
        22: Text Table Partial Match
        23: Text Table No Match
        '''
        sent_len = len(self.src_sent)
        tab_num = len(self.table_names)
        type_mat = []
        for i in range(sent_len):
            type_vec = []
            for j in range(tab_num):
                if self.tab_types[j][i] == 0:
                    type_vec.append(23)
                elif self.tab_types[j][i] == 1:
                    type_vec.append(22)
                else:
                    type_vec.append(21)
            type_mat.append(type_vec)
        return type_mat

    def table_text_type(self):
        '''
        24: Table Text Exact Match
        25: Table Text Partial Match
        26: Table Text No Match
        '''
        tab_num = len(self.table_names)
        sent_len = len(self.src_sent)
        type_mat = []
        for i in range(tab_num):
            type_vec = []
            for j in range(sent_len):
                if self.tab_types[i][j] == 0:
                    type_vec.append(26)
                elif self.tab_types[i][j] == 1:
                    type_vec.append(25)
                else:
                    type_vec.append(24)
            type_mat.append(type_vec)
        return type_mat

    def text_column_type(self):
        '''
        27: Text Column Exact Match
        28: Text Column Partial Match
        29: Text Column No Match
        '''
        sent_len = len(self.src_sent)
        col_num = len(self.column_names)
        type_mat = []
        for i in range(sent_len):
            type_vec = []
            for j in range(col_num):
                if self.col_types[j][i] == 0:
                    type_vec.append(29)
                elif self.col_types[j][i] == 1:
                    type_vec.append(28)
                else:
                    type_vec.append(27)
            type_mat.append(type_vec)
        return type_mat

    def column_text_type(self):
        '''
        30: Column Text Exact Match
        31: Column Text Partial Match
        32: Column Text No Match
        '''
        col_num = len(self.column_names)
        sent_len = len(self.src_sent)
        type_mat = []
        for i in range(col_num):
            type_vec = []
            for j in range(sent_len):
                if self.col_types[i][j] == 0:
                    type_vec.append(32)
                elif self.col_types[i][j] == 1:
                    type_vec.append(31)
                else:
                    type_vec.append(30)
            type_mat.append(type_vec)
        return type_mat


class RatBatch(Batch):
    def __init__(self, examples, grammar, vocab, cuda=False, pretrained_encoder_type="bert"):
        super(RatBatch, self).__init__(examples, grammar, vocab, cuda=cuda)

        self.T = torch.cuda if cuda else torch
        self.cols = [self.get_typed_columns(e) for e in self.examples]
        self.origin_cols = [self.get_typed_columns(e, True) for e in self.examples]
        self.tabs = [e.tab_names for e in self.examples]
        self.origin_tabs = [e.origin_tab_names for e in self.examples]
        self.origin_tabs_cols = [self.get_typed_columns(e, True, True) for e in self.examples]
        self.col_len = [len(col) for col in self.cols]
        self.max_col_num = max(self.col_len)
        self.tab_len = [len(tab) for tab in self.tabs]
        self.max_tab_num = max(self.tab_len)
        self.max_sent_len = max(self.src_sents_len)

        self.enc_len = [q + c + t for q, c,
                        t in zip(self.src_sents_len, self.col_len, self.tab_len)]
        self.max_enc_len = max(self.enc_len)
        self.original_sents = [e.original for e in self.examples]
        if pretrained_encoder_type != "roberta" and hasattr(self.examples[0], 'bert_tokens'):
            self.bert_tokens = [e.bert_tokens for e in self.examples]
            self.bert_tabs_cols = [e.bert_tabs_cols for e in self.examples]

    def get_column_contents(self):

        return [example.col_value for example in self.examples]

    def get_column_types(self):

        return [example.schema._table['exp_column_types'] for example in self.examples]

    def get_typed_columns(self, example, original=False, table=False):
        typed_columns = []
        if table:
            typed_columns = example.origin_tab_names[:]
        if original:
            columns = example.origin_col_names
        else:
            columns = example.col_names
        column_tids = [x[0] for x in example.column_names]
        types = example.schema._table['exp_column_types']
        for col, typ, tid in zip(columns, types, column_tids):
            if original:
                typed_columns.append(typ + ' ' + col)
            else:
                typed_columns.append((tid, typ + ' ' + col))
        return typed_columns

    def _get_relation_matrix(self, example):
        col2col = np.array(example.col2col)
        col2tab = np.array(example.col2tab)
        col2tok = np.array(example.col2tok)

        tab2col = np.array(example.tab2col)
        tab2tab = np.array(example.tab2tab)
        tab2tok = np.array(example.tab2tok)

        tok2col = np.array(example.tok2col)
        tok2tab = np.array(example.tok2tab)
        tok2tok = np.array(example.tok2tok)

        tok_rows = np.hstack([tok2tok, tok2col, tok2tab])
        col_rows = np.hstack([col2tok, col2col, col2tab])
        tab_rows = np.hstack([tab2tok, tab2col, tab2tab])

        return np.vstack([tok_rows, col_rows, tab_rows])

    def get_relation_matrix(self):
        relations = []
        for batch_idx, example in enumerate(self.examples):
            enc_len = self.enc_len[batch_idx]
            relation = np.pad(self._get_relation_matrix(example),
                              ((0, self.max_enc_len - enc_len),), 'constant')
            relations.append(relation)
        return np.stack(relations)

    @cached_property
    def table_mask(self, attention_mask=True):
        max_table_num = max(len(table) for table in self.tabs)
        mask_val = 1 if attention_mask else 0
        table_mask = [[1 - mask_val] * len(table) + [mask_val] * (
            max_table_num - len(table)) for table in self.tabs]
        return self.T.BoolTensor(table_mask)

    @cached_property
    def column_mask(self, attention_mask=True):
        max_column_num = max(len(column) for column in self.cols)
        mask_val = 1 if attention_mask else 0
        column_mask = [[1 - mask_val] * len(column) + [mask_val] * (
            max_column_num - len(column)) for column in self.cols]
        return self.T.BoolTensor(column_mask)

    def get_table_input_tensor(self, pad_lens=True):
        vocab = self.vocab.source
        max_table_num = max(len(table) for table in self.tabs)
        max_table_word_num = max(len(s.split()) + 2 for s in chain.from_iterable(
            table for table in self.tabs))
        table_wids = []
        table_lens = [[len(s.split()) + 2 for s in table] for table in self.tabs]

        if pad_lens:
            for lens in table_lens:
                lens.extend([1] * (max_table_num - len(lens)))

        for table in self.tabs:
            cur_table_wids = []
            for s in table:
                cur_wids = [vocab['<s>']] + \
                    [vocab[token] for token in s.split()] + \
                    [vocab['</s>']] + \
                    [vocab['<pad>']] * (max_table_word_num - len(s.split()) - 2)
                cur_table_wids.append(cur_wids)

            for i in range(max_table_num - len(table)):
                cur_table_wids.append([vocab['<pad>']] * max_table_word_num)

            table_wids.append(cur_table_wids)

        table_wids = self.T.LongTensor(table_wids)

        return table_wids, table_lens

    def get_column_input_tensor(self, pad_lens=True):
        vocab = self.vocab.source
        max_column_num = max(len(column) for column in self.cols)
        max_column_word_num = max(len(s[1].split()) + 2 for s in chain.from_iterable(
            column for column in self.cols))
        column_wids = []
        column_tids = []
        column_lens = [[len(s[1].split()) + 2 for s in column] for column in self.cols]

        if pad_lens:
            for lens in column_lens:
                lens.extend([1] * (max_column_num - len(lens)))

        for column in self.cols:
            cur_column_wids = []
            cur_column_tids = []
            for tid, s in column:
                cur_wids = [vocab['<s>']] + \
                    [vocab[token] for token in s.split()] + \
                    [vocab['</s>']] + \
                    [vocab['<pad>']] * (max_column_word_num - len(s.split()) - 2)
                cur_column_wids.append(cur_wids)
                if tid == -1:
                    raise ValueError
                cur_column_tids.append(tid)

            for i in range(max_column_num - len(column)):
                cur_column_wids.append([vocab['<pad>']] * max_column_word_num)
                cur_column_tids.append(0)

            column_wids.append(cur_column_wids)
            column_tids.append(cur_column_tids)

        column_wids = self.T.LongTensor(column_wids)
        column_tids = self.T.LongTensor(column_tids)

        return column_wids, column_tids, column_lens

    def get_column_ids_tensor(self, pad_lens=True):
        max_column_num = max(len(column) for column in self.cols)

        column_tids = []
        column_lens = [[len(s[1].split()) + 2 for s in column] for column in self.cols]

        if pad_lens:
            for lens in column_lens:
                lens.extend([1] * (max_column_num - len(lens)))

        for column in self.cols:
            cur_column_tids = []
            for tid, s in column:
                if tid == -1:
                    raise ValueError
                cur_column_tids.append(tid)

            for i in range(max_column_num - len(column)):
                cur_column_tids.append(0)

            column_tids.append(cur_column_tids)

        column_tids = self.T.LongTensor(column_tids)

        return column_tids

    @cached_property
    def src_token_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.src_sents_len, cuda=self.cuda)
