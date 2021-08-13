# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pickle
import json
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
import os
from semparser.modules.alanschema.scripts.generate_schema_with_db_meta import update_schemas_with_meta

from semparser.common import registry
from semparser.modules.preprocessing.nlp_preprocessor import AlanNLPPreprocessor
from semparser.modules.preprocessing.schema_linker.rat_schema_linker import RatSchemaLinker
from semparser.modules.preprocessing.text_normalizers.basic_normalizers import QuoteNormalizer, IDNormalizer
from semparser.modules.alanschema.schema_with_meta import get_schemas_with_meta_from_json, \
    load_columns_with_meta, load_tables_with_meta
from semparser.data.rat import RatExample

from semparser.modules.semantic_parser.asdl.action_info import ActionInfo
from semparser.modules.semantic_parser.asdl.spider.spider_transition_system import sql_query_to_asdl_ast, \
    asdl_ast_to_sql_query
from semparser.modules.semantic_parser.asdl.spider.spider_hypothesis import SpiderHypothesis
from semparser.modules.semantic_parser.evaluator.spider_evaluator import SpiderEvaluator, rebuild_sql_val, \
    DISABLE_VALUE, DISABLE_DISTINCT, build_foreign_key_map, rebuild_sql_col, build_valid_col_units, WHERE_OPS, is_sql_valid
from semparser.modules.semantic_parser.preprocessor.process_spider_sql import Schema, get_sql
from semparser.modules.semantic_parser.preprocessor.errata import errata

WRONG_QUERY = ['SELECT T1.company_name FROM Third_Party_Companies AS T1 JOIN Maintenance_Contracts AS T2 ON T1.compa'
               'ny_id  =  T2.maintenance_contract_company_id JOIN Ref_Company_Types AS T3 ON T1.company_type_code  =  '
               'T3.company_type_code ORDER BY T2.contract_end_date DESC LIMIT 1']

def get_action_infos(src_query, tgt_actions, schema):
    action_infos = []
    hyp = SpiderHypothesis(schema)
    for t, action in enumerate(tgt_actions):
        action_info = ActionInfo(action)
        action_info.t = t
        if hyp.frontier_node:
            action_info.parent_t = hyp.frontier_node.created_time
            action_info.frontier_prod = hyp.frontier_node.production
            action_info.frontier_field = hyp.frontier_field.field
        hyp.apply_action(action)
        action_infos.append(action_info)

    return action_infos

def load_dataset(transition_system, dataset_file, schemas_with_meta, tables, schema_linker,
                 drop_invalid=True, for_inference=False):
    evaluator = SpiderEvaluator()
    len_stats = defaultdict(int)
    examples = []
    err_num = 0

    with open(dataset_file, 'r') as f:
        data = json.load(f)

    print("processing file: {}".format(dataset_file))
    print("length: {}".format(len(data)))

    alan_nlp_preprocessor = AlanNLPPreprocessor(
        text_normalizers=[QuoteNormalizer(), IDNormalizer()],
        entity_recognizer_type=None,
        temporal_tagger_type="rule",
        tag_and_lemma_type='wordnet')

    for idx, entry in tqdm(enumerate(data), desc="preprocess", total=len(data)):
        db_id = entry['db_id']
        if for_inference:
            query = None
        else:
            query = entry['query']
        # check if the current example contains uri for backward compatibility
        uri = entry['uri'] if 'uri' in entry else None

        if query in WRONG_QUERY:
            continue
        if query in errata:
            query = errata[query]
        question_toks = entry['question_toks']
        schema = Schema(schemas_with_meta[db_id], tables[db_id])
        columns_with_meta = load_columns_with_meta(schema._table)
        tables_with_meta = load_tables_with_meta(schema._table)

        # set schema in schema linker
        schema_linker.set_schema(tables_with_meta, columns_with_meta, alan_nlp_preprocessor)

        # process document
        nlp_document = alan_nlp_preprocessor.preprocessing(" ".join(question_toks))

        # get schema linking result
        question_toks, original_toks, tab_names, col_names, \
            origin_tab_names, origin_col_names, tab_types, col_types = schema_linker.linking(nlp_document)

        if not for_inference:
            try:
                spider_sql = get_sql(schema, query)
            except Exception:
                print("Cannot preprocess:", query)
                continue
            spider_sql_copy = deepcopy(spider_sql)
            _spider_sql_copy = deepcopy(spider_sql)
            try:
                asdl_ast = sql_query_to_asdl_ast(spider_sql, transition_system.grammar, schema)
            except KeyError as e:
                print(e)
                print("The current ASDL cannot process:", query)
                continue
            asdl_ast.sanity_check()
            actions = transition_system.get_actions(asdl_ast)
            tgt_action_infos = get_action_infos(question_toks, actions, schema)

            hyp = SpiderHypothesis(schema)
            for action, action_info in zip(actions, tgt_action_infos):
                assert action == action_info.action
                hyp.apply_action(action)

            kmap = build_foreign_key_map(schema._table)
            valid_col_units1 = build_valid_col_units(spider_sql_copy['from']['table_units'], schema)
            spider_sql_copy = rebuild_sql_val(spider_sql_copy)
            spider_sql_copy = rebuild_sql_col(valid_col_units1, spider_sql_copy, kmap)

            recon_hyp_query = asdl_ast_to_sql_query(hyp.tree, schema)
            try:
                assert is_sql_valid(recon_hyp_query, schema)
            except Exception:

                print(' '.join(question_toks))
                print(query)
            valid_col_units2 = build_valid_col_units(recon_hyp_query['from']['table_units'], schema)
            recon_hyp_query = rebuild_sql_val(recon_hyp_query)
            recon_hyp_query = rebuild_sql_col(valid_col_units2, recon_hyp_query, kmap)
            try:
                assert evaluator.eval_exact_match(recon_hyp_query, spider_sql_copy), \
                    "\n" + repr(spider_sql_copy) + "\n" + repr(recon_hyp_query)
            except Exception:
                err_num += 1
                if drop_invalid:
                    continue
        else:
            _spider_sql_copy = None
            tgt_action_infos = []
            asdl_ast = None

        example = RatExample(question=question_toks,
                             original=original_toks,
                             tab_names=tab_names,
                             col_names=col_names,
                             origin_tab_names=origin_tab_names,
                             origin_col_names=origin_col_names,
                             tab_types=tab_types,
                             col_types=col_types,
                             sql=_spider_sql_copy,
                             schema=schema,
                             tgt_actions=tgt_action_infos,
                             tgt_code=query,
                             tgt_ast=asdl_ast,
                             idx=idx,
                             uri=uri,
                             db_id=db_id)

        if 'attributes' in entry:
            example.attributes = entry['attributes']

        total_len = len(question_toks) + len(tab_names) + len(col_names)
        len_stats[total_len // 100] += 1
        if total_len <= 200:
            examples.append(example)
        elif not drop_invalid:
            examples.append(example)

    print(len_stats)
    print(err_num)
    return examples


@registry.register('preprocessor', 'rat_new_sl')
def prepare_data(raw_train_data, raw_dev_data, raw_schema, schema_with_meta, database_folder,
                 train_data_path, dev_data_path, concept_net_folder, wordnet_synonyms_for_table_and_column,
                 use_temporal_result, use_entity_result, use_alter_names, use_db_value, token_type_norm,
                 transition_system, for_inference=False):
    if not os.path.exists(schema_with_meta):
        print('Creating schema with meta...')
        schema_with_db_meta = update_schemas_with_meta(raw_schema, database_folder)
        with open(schema_with_meta, 'w') as f:
            json.dump(schema_with_db_meta, f, indent=4)

    schemas, db_names, tables = get_schemas_with_meta_from_json(schema_with_meta)

    # create schema_linker
    schema_linker = RatSchemaLinker(
        concept_net_folder,
        wordnet_synonyms_for_table_and_column,
        use_temporal_result,
        use_entity_result,
        use_alter_names,
        use_db_value,
        token_type_norm)

    if not for_inference:
        train_data = load_dataset(transition_system, raw_train_data, schemas, tables, schema_linker)
    dev_data = load_dataset(transition_system, raw_dev_data, schemas, tables, schema_linker, False, for_inference)

    if not for_inference:
        with open(train_data_path, "wb") as f:
            pickle.dump(train_data, f)
    with open(dev_data_path, "wb") as f:
        pickle.dump(dev_data, f)
