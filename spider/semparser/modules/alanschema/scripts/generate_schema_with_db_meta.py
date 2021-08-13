# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""script to summarize spider db content, it will later be used for schema linking"""
import os
import json
import argparse
from tqdm import tqdm
from typing import List, Dict
import logging
import sqlite3
from semparser.modules.preprocessing.nlp_preprocessor import AlanNLPPreprocessor
from semparser.modules.preprocessing.text_normalizers.basic_normalizers import TextNormalizer, QuoteNormalizer, IDNormalizer
from semparser.modules.preprocessing.nlp_document import normalize_doc

import numpy as np


def get_token_bodies(nlp_tokens):
    """
    Get token body based on the linker setting
    """

    tokens = []
    for t in nlp_tokens:
        if t.tag.startswith("V") or t.tag.startswith("N"):
            tokens.append(t.lemmatized_body.lower())
        else:
            tokens.append(t.body.lower())
    return tokens


data_preprocessor = AlanNLPPreprocessor(
    text_normalizers=[IDNormalizer()],
    entity_recognizer_type=None,
    temporal_tagger_type="rule",
    tag_and_lemma_type='wordnet')


def normalize_str_value(value):
    if isinstance(value, str):
        return value.lower()
    try:
        return str(value).lower()
    except:
        return ""


def extract_column_meta(col_values, alter_names, col_type):

    # some cleanup
    col_values = [n for n in col_values if n]
    if col_type == "number":
        col_values = [n for n in col_values if not isinstance(n, str)]
    categorical_values = None
    value_range = None
    meta = {}
    if col_values:
        if col_type == "text":
            col_values = [normalize_str_value(val) for val in col_values]
            col_values = [v for v in col_values if v != ""]
            distinct_values = set(col_values)
            distinct_values = [" ".join(get_token_bodies(
                data_preprocessor.preprocessing(v).nlp_tokens)) for v in distinct_values]
            categorical_values = list([str(v) for v in distinct_values])
        elif col_type == "number":
            col_values = np.array(col_values)
            mean = col_values.mean()
            var = col_values.var()
            value_range = {"mean": mean, "var": var}
    meta = {
        "type": col_type,
        "alter_names": alter_names,
        "categorical_options": categorical_values,
        "range": value_range
    }
    return meta


def _execute_sql(sqlite_path, sql):
    con = sqlite3.connect(sqlite_path)
    con.text_factory = str
    cur = con.cursor()
    cur.execute(sql)
    ret = cur.fetchall()
    con.close()
    return ret


def get_db_sqlite_path(database_folder, db_id):
    return os.path.join(database_folder, db_id, '{}.sqlite'.format(db_id))


def get_db_column_values(database_folder, db_id, table_name, col_name):
    sqlite_path = get_db_sqlite_path(database_folder, db_id)
    quoted_col_name = "\"{}\"".format(col_name)
    sql = "SELECT {} from {}".format(quoted_col_name, table_name)
    col_values = []
    try:
        col_values = [item[0] for item in _execute_sql(sqlite_path, sql)]
    except Exception as e:
        logging.warning("can't do {} on {} with error message {}".format(sql, db_id, e))
    return col_values


def update_schemas_with_meta(table_fpath, database_folder):
    with open(table_fpath, 'r') as f:
        schemas = json.load(f)
    schemas_with_db_meta = []
    for schema in tqdm(schemas):
        db_id = schema['db_id']
        orig_table_names = [table.lower() for table in schema['table_names_original']]
        orig_column_names = [(i, col.lower())
                             for i, col in schema['column_names_original'][1:]]
        column_types = [col_type.lower() for col_type in schema['column_types'][1:]]

        # column meta
        # for the first * column, meta is default None
        column_metas = [None]
        if 'column_alter_names' in schema:
            column_alter_names = schema['column_alter_names'].copy()
        else:
            column_alter_names = [[]] * len(column_types)
        for (tab_idx, col_name), col_type, alter_names in zip(orig_column_names, column_types, column_alter_names):
            db_values = get_db_column_values(
                database_folder, db_id, orig_table_names[tab_idx], col_name)

            column_meta = extract_column_meta(db_values, alter_names, col_type)
            column_metas.append(column_meta)
        schema["column_metas"] = column_metas

        # table meta
        table_metas = []
        if 'table_alter_names' in schema:
            table_alter_names = schema['table_alter_names'].copy()
        else:
            table_alter_names = [[]] * len(orig_table_names)
        for alter_names in table_alter_names:
            table_metas.append({'alter_names': alter_names})
        schema["table_metas"] = table_metas

        schemas_with_db_meta.append(schema)
    return schemas_with_db_meta


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--data_folder",
                            type=str,
                            required=True,
                            help="path to tables.json file")
    arg_parser.add_argument("--database_folder",
                            type=str,
                            required=True,
                            help="path to sqlite files")
    arg_parser.add_argument("--output_fname",
                            type=str,
                            help="fname to save schema with db meta",
                            default="tables_with_db_meta.json")

    args = arg_parser.parse_args()
    table_fpath = os.path.join(args.data_folder, "tables.json")
    database_folder = args.database_folder
    schemas_with_db_meta = update_schemas_with_meta(
        table_fpath, database_folder)
    output_fpath = os.path.join(args.data_folder, args.output_fname)
    with open(output_fpath, 'w') as f:
        json.dump(schemas_with_db_meta, f, indent=4)
