# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pickle
import json
from collections import defaultdict


class ColumnWithMeta:
    def __init__(
        self,
        tab_index,
        name,
        alter_names = [],
        value_type = None,
        value_range = None,
        categorical_options = None):
        self.tab_index = tab_index
        self.name = name
        self.value_type = value_type
        self.value_range = value_range
        self.alter_names = alter_names
        self.categorical_options = categorical_options 


class TableWithMeta:
    def __init__(
        self,
        name,
        alter_names = []
    ):
        self.name = name
        self.alter_names = alter_names

def load_columns_with_meta(schema):
    columns_with_meta = []
    column_type = None
    for (tab_index, name), meta in zip(schema['exp_column_names'], schema['exp_column_metas']):
        column = ColumnWithMeta(tab_index, name)
        if meta:
            if "categorical_options" in meta:
                if meta["categorical_options"]:
                    column.categorical_options = [str(t) for t in meta["categorical_options"]]
                else:
                    column.categorical_options = None
            if "type" in meta:
                column.value_type = meta["type"]
            else:
                column.value_type = "text"
            if "range" in meta:
                column.value_range = meta["range"]
            if "alter_names" in meta:
                column.alter_names = meta["alter_names"]
        
        columns_with_meta.append(column)
    return columns_with_meta


def load_tables_with_meta(schema):
    tables_with_meta = []
    for table_name, meta in zip(schema["exp_table_names"], schema["exp_table_metas"]):
        if meta and 'alter_names' in meta:
            table = TableWithMeta(table_name, meta['alter_names'])
        else:
            table = TableWithMeta(table_name, [])
        tables_with_meta.append(table)
        
    return tables_with_meta  

def get_schemas_with_meta_from_json(fpath):
    with open(fpath) as f:
        data = json.load(f)
    db_names = [db['db_id'] for db in data]

    tables = {}
    schemas = {}
    for db in data:
        db_id = db['db_id']
        schema = {}  # {'table': [col.lower, ..., ]} * -> __all__
        column_names = db['column_names']
        column_names = [(x[0], x[1].lower()) for x in column_names]
        table_names = db['table_names']
        table_names = [x.lower() for x in table_names]
        column_names_original = db['column_names_original']
        column_names_original = [(x[0], x[1].lower()) for x in column_names_original]
        table_names_original = db['table_names_original']
        table_names_original = [x.lower() for x in table_names_original]
        column_types = db['column_types']
        foreign_keys = db['foreign_keys']
        primary_keys = db['primary_keys']
        column_metas = db['column_metas']
        table_metas = db['table_metas']
        col_ori_to_exp = {}
        col_exp_to_ori = {}
        exp_column_names = []
        exp_column_names_original = []
        exp_table_names = table_names + ['*']
        exp_table_names_original = table_names_original + ['*']
        exp_table_metas = table_metas + [None]
        cur = 0
        for idx, (c1, c2) in enumerate(zip(column_names, column_names_original)):
            if idx == 0:
                exp_column_names.append((-1, c1[1]))
                exp_column_names_original.append((-1, c2[1]))
                col_exp_to_ori[cur] = 0
                cur += 1
                for i in range(len(table_names_original)):
                    exp_column_names.append((i, c1[1]))
                    exp_column_names_original.append((i, c2[1]))
                    col_exp_to_ori[cur] = 0
                    cur += 1
            else:
                exp_column_names.append(c1)
                exp_column_names_original.append(c2)
                col_exp_to_ori[cur] = idx
                col_ori_to_exp[idx] = cur
                cur += 1
        exp_column_types = [column_types[col_exp_to_ori[i]] for i in range(len(exp_column_names))]
        exp_column_metas = [column_metas[col_exp_to_ori[i]] for i in range(len(exp_column_names))]
        exp_foreign_keys = [[col_ori_to_exp[x], col_ori_to_exp[y]] for x, y in foreign_keys]
        exp_primary_keys = [col_ori_to_exp[x] for x in primary_keys]

        tables[db_id] = {'column_names': column_names, 'table_names': table_names,
                         'column_names_original': column_names_original, 'table_names_original': table_names_original,
                         'foreign_keys': foreign_keys, 'primary_keys': primary_keys, 'column_types': column_types,
                         'exp_column_names': exp_column_names, 'exp_column_names_original': exp_column_names_original,
                         'exp_table_names': exp_table_names, 'exp_table_names_original': exp_table_names_original,
                         'col_ori_to_exp': col_ori_to_exp, 'col_exp_to_ori': col_exp_to_ori,
                         'exp_foreign_keys': exp_foreign_keys, 'exp_primary_keys': exp_primary_keys,
                         'exp_column_types': exp_column_types, 'exp_column_metas': exp_column_metas, 'exp_table_metas': exp_table_metas}
        for i, tabn in enumerate(table_names_original):
            table = str(tabn.lower())
            cols = [str(col.lower()) for td, col in column_names_original if td == i]
            schema[table] = cols
        schemas[db_id] = schema
    return schemas, db_names, tables
