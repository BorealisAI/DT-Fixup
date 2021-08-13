# Copyright (c) 2020-present, Royal Bank of Canada.
# Copyright (c) 2018-present, Tao Yu
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#################################################################################################
# Code is based on the Spider (https://arxiv.org/abs/1809.08887) implementation
# from https://github.com/taoyds/spider by Tao Yu
#################################################################################################


################################
# val: number(float)/string(str)/sql(dict)
# col_unit: (agg_id, col_id, isDistinct(bool))
# val_unit: (unit_op, col_unit1, col_unit2)
# table_unit: (table_type, col_unit/sql)
# cond_unit: (not_op, op_id, val_unit, val1, val2)
# condition: [cond_unit1, 'and'/'or', cond_unit2, ...]
# sql {
#   'select': (isDistinct(bool), [(agg_id, val_unit), (agg_id, val_unit), ...])
#   'from': {'table_units': [table_unit1, table_unit2, ...], 'conds': condition}
#   'where': condition
#   'groupBy': [col_unit1, col_unit2, ...]
#   'orderBy': ('asc'/'desc', [val_unit1, val_unit2, ...])
#   'having': condition
#   'limit': None/limit value
#   'intersect': None/sql
#   'except': None/sql
#   'union': None/sql
# }
################################

import os
import json
import sqlite3
import argparse
import logging
import pickle
from copy import deepcopy
import difflib
import traceback

from semparser.common import registry
from semparser.common.utils import print_dict

from semparser.modules.semantic_parser.preprocessor.process_spider_sql import get_sql, get_schema
from semparser.modules.semantic_parser.asdl.spider.spider_hypothesis import SpiderDecodeHypothesis
from semparser.modules.semantic_parser.inference.spider_ast import SpiderAST

# Flag to disable value evaluation
DISABLE_VALUE = True
# Flag to disable distinct in select evaluation
DISABLE_DISTINCT = True

CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order',
                   'limit', 'intersect', 'union', 'except')
JOIN_KEYWORDS = ('join', 'on', 'as')

WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
TABLE_TYPE = {
    'sql': "sql",
    'table_unit': "table_unit",
}

COND_OPS = ('and', 'or')
SQL_OPS = ('intersect', 'union', 'except')
ORDER_OPS = ('desc', 'asc')

HARDNESS = {
    "component1": ('where', 'group', 'order', 'limit', 'join', 'or', 'like'),
    "component2": ('except', 'union', 'intersect')
}


class Schema:
    def __init__(self, schema):
        self._schema = schema
        self._idMap = self._map(self._schema)

    @property
    def schema(self):
        return self._schema

    @property
    def idMap(self):
        return self._idMap

    def _map(self, schema):
        idMap = {'*': "__all__"}
        id = 1
        for key, vals in schema.iteritems():
            for val in vals:
                idMap[key.lower() + "." + val.lower()] = "__" + \
                    key.lower() + "." + val.lower() + "__"
                id += 1

        for key in schema:
            idMap[key.lower()] = "__" + key.lower() + "__"

        return idMap


def condition_has_or(conds):
    return 'or' in conds[1::2]


def condition_has_like(conds):
    return WHERE_OPS.index('like') in [cond_unit[1] for cond_unit in conds[::2]]


def condition_has_sql(conds):
    for cond_unit in conds[::2]:
        val1, val2 = cond_unit[3], cond_unit[4]
        if val1 is not None and type(val1) is dict:
            return True
        if val2 is not None and type(val2) is dict:
            return True
    return False


def val_has_op(val_unit):
    return val_unit[0] != UNIT_OPS.index('none')


def has_agg(unit):
    return unit[0] != AGG_OPS.index('none')


def accuracy(count, total):
    if count == total:
        return 1
    return 0


def recall(count, total):
    if count == total:
        return 1
    return 0


def F1(acc, rec):
    if (acc + rec) == 0:
        return 0
    return (2. * acc * rec) / (acc + rec)


def get_scores(count, pred_total, label_total):
    if pred_total != label_total:
        return 0, 0, 0
    elif count == pred_total:
        return 1, 1, 1
    return 0, 0, 0


def eval_sel(pred, label):
    pred_sel = pred['select'][1]
    label_sel = [x for x in label['select'][1]]
    label_wo_agg = [unit[1] for unit in label_sel]
    pred_total = len(pred_sel)
    label_total = len(label_sel)
    cnt = 0
    cnt_wo_agg = 0

    for unit in pred_sel:
        if unit in label_sel:
            cnt += 1
            label_sel.remove(unit)
        if unit[1] in label_wo_agg:
            cnt_wo_agg += 1
            label_wo_agg.remove(unit[1])

    return label_total, pred_total, cnt, cnt_wo_agg


def eval_where(pred, label):
    pred_conds = [unit for unit in pred['where'][::2]]
    label_conds = [unit for unit in label['where'][::2]]
    label_wo_agg = [unit[2] for unit in label_conds]
    pred_total = len(pred_conds)
    label_total = len(label_conds)
    cnt = 0
    cnt_wo_agg = 0

    for unit in pred_conds:
        if unit in label_conds:
            cnt += 1
            label_conds.remove(unit)
        if unit[2] in label_wo_agg:
            cnt_wo_agg += 1
            label_wo_agg.remove(unit[2])

    return label_total, pred_total, cnt, cnt_wo_agg


def eval_group(pred, label):
    pred_cols = [unit[1] for unit in pred['groupBy']]
    label_cols = [unit[1] for unit in label['groupBy']]
    pred_total = len(pred_cols)
    label_total = len(label_cols)
    cnt = 0
    pred_cols = [pred for pred in pred_cols]
    label_cols = [label for label in label_cols]
    for col in pred_cols:
        if col in label_cols:
            cnt += 1
            label_cols.remove(col)
    return label_total, pred_total, cnt


def eval_having(pred, label):
    pred_total = label_total = cnt = 0
    if len(pred['groupBy']) > 0:
        pred_total = 1
    if len(label['groupBy']) > 0:
        label_total = 1

    pred_cols = [unit[1] for unit in pred['groupBy']]
    label_cols = [unit[1] for unit in label['groupBy']]
    if pred_total == label_total == 1 \
            and pred_cols == label_cols \
            and pred['having'] == label['having']:
        cnt = 1

    return label_total, pred_total, cnt


def eval_order(pred, label):
    pred_total = label_total = cnt = 0
    if len(pred['orderBy']) > 0:
        pred_total = 1
    if len(label['orderBy']) > 0:
        label_total = 1
    if len(label['orderBy']) > 0 and pred['orderBy'] == label['orderBy'] and (
            (pred['limit'] is None and label['limit'] is None) or (
                pred['limit'] is not None and label['limit'] is not None)):
        cnt = 1
    return label_total, pred_total, cnt


def eval_and_or(pred, label):
    pred_ao = pred['where'][1::2]
    label_ao = label['where'][1::2]
    pred_ao = set(pred_ao)
    label_ao = set(label_ao)

    if pred_ao == label_ao:
        return 1, 1, 1
    return len(pred_ao), len(label_ao), 0


def get_nestedSQL(sql):
    nested = []
    for cond_unit in sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]:
        if type(cond_unit[3]) is dict:
            nested.append(cond_unit[3])
        if type(cond_unit[4]) is dict:
            nested.append(cond_unit[4])
    if sql['intersect'] is not None:
        nested.append(sql['intersect'])
    if sql['except'] is not None:
        nested.append(sql['except'])
    if sql['union'] is not None:
        nested.append(sql['union'])
    return nested


def eval_nested(pred, label):
    label_total = 0
    pred_total = 0
    cnt = 0
    if pred is not None:
        pred_total += 1
    if label is not None:
        label_total += 1
    if pred is not None and label is not None:
        cnt += SpiderEvaluator().eval_exact_match(pred, label)
    return label_total, pred_total, cnt


def eval_IUEN(pred, label):
    lt1, pt1, cnt1 = eval_nested(pred['intersect'], label['intersect'])
    lt2, pt2, cnt2 = eval_nested(pred['except'], label['except'])
    lt3, pt3, cnt3 = eval_nested(pred['union'], label['union'])
    label_total = lt1 + lt2 + lt3
    pred_total = pt1 + pt2 + pt3
    cnt = cnt1 + cnt2 + cnt3
    return label_total, pred_total, cnt


def get_keywords(sql):
    res = set()
    if len(sql['where']) > 0:
        res.add('where')
    if len(sql['groupBy']) > 0:
        res.add('group')
    if len(sql['having']) > 0:
        res.add('having')
    if len(sql['orderBy']) > 0:
        res.add(sql['orderBy'][0])
        res.add('order')
    if sql['limit'] is not None:
        res.add('limit')
    if sql['except'] is not None:
        res.add('except')
    if sql['union'] is not None:
        res.add('union')
    if sql['intersect'] is not None:
        res.add('intersect')

    # or keyword
    ao = sql['from']['conds'][1::2] + sql['where'][1::2] + sql['having'][1::2]
    if len([token for token in ao if token == 'or']) > 0:
        res.add('or')

    cond_units = sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]
    # not keyword
    if len([cond_unit for cond_unit in cond_units if cond_unit[0]]) > 0:
        res.add('not')

    # in keyword
    if len([cond_unit for cond_unit in cond_units if cond_unit[1] == WHERE_OPS.index('in')]) > 0:
        res.add('in')

    # like keyword
    if len([cond_unit for cond_unit in cond_units if cond_unit[1] == WHERE_OPS.index('like')]) > 0:
        res.add('like')

    return res


def eval_keywords(pred, label):
    pred_keywords = get_keywords(pred)
    label_keywords = get_keywords(label)
    pred_total = len(pred_keywords)
    label_total = len(label_keywords)
    cnt = 0

    for k in pred_keywords:
        if k in label_keywords:
            cnt += 1
    return label_total, pred_total, cnt


def count_agg(units):
    return len([unit for unit in units if has_agg(unit)])


def count_component1(sql):
    count = 0
    if len(sql['where']) > 0:
        count += 1
    if len(sql['groupBy']) > 0:
        count += 1
    if len(sql['orderBy']) > 0:
        count += 1
    if sql['limit'] is not None:
        count += 1
    if len(sql['from']['table_units']) > 0:  # JOIN
        count += len(sql['from']['table_units']) - 1

    ao = sql['from']['conds'][1::2] + sql['where'][1::2] + sql['having'][1::2]
    count += len([token for token in ao if token == 'or'])
    cond_units = sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]
    count += len([cond_unit for cond_unit in cond_units if cond_unit[1] == WHERE_OPS.index('like')])

    return count


def count_component2(sql):
    nested = get_nestedSQL(sql)
    return len(nested)


def count_others(sql):
    count = 0
    # number of aggregation
    agg_count = count_agg(sql['select'][1])
    agg_count += count_agg(sql['where'][::2])
    agg_count += count_agg(sql['groupBy'])
    if len(sql['orderBy']) > 0:
        agg_count += count_agg([unit[1] for unit in sql['orderBy'][1] if unit[1]] + [
            unit[2] for unit in sql['orderBy'][1] if unit[2]])
    agg_count += count_agg(sql['having'])
    if agg_count > 1:
        count += 1

    # number of select columns
    if len(sql['select'][1]) > 1:
        count += 1

    # number of where conditions
    if len(sql['where']) > 1:
        count += 1

    # number of group by clauses
    if len(sql['groupBy']) > 1:
        count += 1

    return count


class SpiderEvaluator:
    """A simple evaluator"""

    def __init__(self):
        self.partial_scores = None

    def eval_hardness(self, sql):
        count_comp1_ = count_component1(sql)
        count_comp2_ = count_component2(sql)
        count_others_ = count_others(sql)

        if count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ == 0:
            return "easy"
        elif (count_others_ <= 2 and count_comp1_ <= 1 and count_comp2_ == 0) or \
                (count_comp1_ <= 2 and count_others_ < 2 and count_comp2_ == 0):
            return "medium"
        elif (count_others_ > 2 and count_comp1_ <= 2 and count_comp2_ == 0) or \
                (2 < count_comp1_ <= 3 and count_others_ <= 2 and count_comp2_ == 0) or \
                (count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ <= 1):
            return "hard"
        else:
            return "extra"

    def eval_exact_match(self, pred, label):
        partial_scores = self.eval_partial_match(pred, label)
        self.partial_scores = partial_scores

        for x, score in partial_scores.items():
            if score['f1'] != 1:
                return 0
        if len(label['from']['table_units']) > 0:
            label_tables = sorted(label['from']['table_units'])
            pred_tables = sorted(pred['from']['table_units'])
            return label_tables == pred_tables
        return 1

    def eval_partial_match(self, pred, label):
        res = {}

        label_total, pred_total, cnt, cnt_wo_agg = eval_sel(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['select'] = {'acc': acc, 'rec': rec, 'f1': f1,
                         'label_total': label_total, 'pred_total': pred_total}
        acc, rec, f1 = get_scores(cnt_wo_agg, pred_total, label_total)
        res['select(no AGG)'] = {'acc': acc, 'rec': rec, 'f1': f1,
                                 'label_total': label_total, 'pred_total': pred_total}

        label_total, pred_total, cnt, cnt_wo_agg = eval_where(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['where'] = {'acc': acc, 'rec': rec, 'f1': f1,
                        'label_total': label_total, 'pred_total': pred_total}
        acc, rec, f1 = get_scores(cnt_wo_agg, pred_total, label_total)
        res['where(no OP)'] = {'acc': acc, 'rec': rec, 'f1': f1,
                               'label_total': label_total, 'pred_total': pred_total}

        label_total, pred_total, cnt = eval_group(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['group(no Having)'] = {'acc': acc, 'rec': rec, 'f1': f1, 'label_total': label_total,
                                   'pred_total': pred_total}

        label_total, pred_total, cnt = eval_having(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['group'] = {'acc': acc, 'rec': rec, 'f1': f1,
                        'label_total': label_total, 'pred_total': pred_total}

        label_total, pred_total, cnt = eval_order(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['order'] = {'acc': acc, 'rec': rec, 'f1': f1,
                        'label_total': label_total, 'pred_total': pred_total}

        label_total, pred_total, cnt = eval_and_or(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['and/or'] = {'acc': acc, 'rec': rec, 'f1': f1,
                         'label_total': label_total, 'pred_total': pred_total}

        label_total, pred_total, cnt = eval_IUEN(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['IUEN'] = {'acc': acc, 'rec': rec, 'f1': f1,
                       'label_total': label_total, 'pred_total': pred_total}

        label_total, pred_total, cnt = eval_keywords(pred, label)
        acc, rec, f1 = get_scores(cnt, pred_total, label_total)
        res['keywords'] = {'acc': acc, 'rec': rec, 'f1': f1,
                           'label_total': label_total, 'pred_total': pred_total}

        return res


def isValidSQL(sql, db):
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
    except Exception:
        return False
    return True


def print_scores(scores, etype, p_func=print):
    levels = ['easy', 'medium', 'hard', 'extra', 'all']
    partial_types = ['select', 'select(no AGG)', 'where', 'where(no OP)', 'group(no Having)',
                     'group', 'order', 'and/or', 'IUEN', 'keywords']

    p_func("{:20} {:20} {:20} {:20} {:20} {:20}".format("", *levels))
    counts = [scores[level]['count'] for level in levels]
    p_func("{:20} {:<20d} {:<20d} {:<20d} {:<20d} {:<20d}".format("count", *counts))

    if etype in ["all", "exec"]:
        p_func('=====================   EXECUTION ACCURACY     =====================')
        this_scores = [scores[level]['exec'] for level in levels]
        p_func("{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format(
            "execution", *this_scores))

    if etype in ["all", "match"]:
        p_func('\n====================== EXACT MATCHING ACCURACY =====================')
        exact_scores = [scores[level]['exact'] for level in levels]
        p_func("{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format(
            "exact match", *exact_scores))

        if 'exact_alan' in scores['all']:
            exact_scores_alan = [scores[level]['exact_alan'] for level in levels]
            p_func("{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format(
                "Alan exact match", *exact_scores_alan))

        p_func('\n---------------------PARTIAL MATCHING ACCURACY----------------------')
        for type_ in partial_types:
            this_scores = [scores[level]['partial'][type_]['acc'] for level in levels]
            p_func("{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format(type_, *this_scores))

        p_func('---------------------- PARTIAL MATCHING RECALL ----------------------')
        for type_ in partial_types:
            this_scores = [scores[level]['partial'][type_]['rec'] for level in levels]
            p_func("{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format(type_, *this_scores))

        p_func('---------------------- PARTIAL MATCHING F1 --------------------------')
        for type_ in partial_types:
            this_scores = [scores[level]['partial'][type_]['f1'] for level in levels]
            p_func("{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format(type_, *this_scores))


def evaluate(gold, predict, db_dir, etype, kmaps):
    with open(gold) as f:
        glist = [l.strip().split('\t') for l in f.readlines() if len(l.strip()) > 0]

    with open(predict) as f:
        plist = [l.strip().split('\t') for l in f.readlines() if len(l.strip()) > 0]
    evaluator = SpiderEvaluator()

    levels = ['easy', 'medium', 'hard', 'extra', 'all']
    partial_types = ['select', 'select(no AGG)', 'where', 'where(no OP)', 'group(no Having)',
                     'group', 'order', 'and/or', 'IUEN', 'keywords']
    entries = []
    scores = {}

    for level in levels:
        scores[level] = {'count': 0, 'partial': {}, 'exact': 0.}
        scores[level]['exec'] = 0
        for type_ in partial_types:
            scores[level]['partial'][type_] = {'acc': 0.,
                                               'rec': 0., 'f1': 0., 'acc_count': 0, 'rec_count': 0}

    eval_err_num = 0
    for p, g in zip(plist, glist):
        p_str = p[0]
        g_str, db = g
        db_name = db
        db = os.path.join(db_dir, db, db + ".sqlite")
        schema = Schema(get_schema(db))
        g_sql = get_sql(schema, g_str)
        hardness = evaluator.eval_hardness(g_sql)
        scores[hardness]['count'] += 1
        scores['all']['count'] += 1

        try:
            p_sql = get_sql(schema, p_str)
        except Exception:
            # If p_sql is not valid, then we will use an empty sql to evaluate with the correct sql
            p_sql = {
                "except": None,
                "from": {
                    "conds": [],
                    "table_units": []
                },
                "groupBy": [],
                "having": [],
                "intersect": None,
                "limit": None,
                "orderBy": [],
                "select": [
                    False,
                    []
                ],
                "union": None,
                "where": []
            }
            eval_err_num += 1
            print("eval_err_num:{}".format(eval_err_num))

        # rebuild sql for value evaluation
        kmap = kmaps[db_name]
        g_valid_col_units = build_valid_col_units(g_sql['from']['table_units'], schema)
        g_sql = rebuild_sql_val(g_sql)
        g_sql = rebuild_sql_col(g_valid_col_units, g_sql, kmap)
        p_valid_col_units = build_valid_col_units(p_sql['from']['table_units'], schema)
        p_sql = rebuild_sql_val(p_sql)
        p_sql = rebuild_sql_col(p_valid_col_units, p_sql, kmap)

        if etype in ["all", "exec"]:
            exec_score = eval_exec_match(db, p_str, g_str, p_sql, g_sql)
            if exec_score:
                scores[hardness]['exec'] += 1

        if etype in ["all", "match"]:
            exact_score = evaluator.eval_exact_match(p_sql, g_sql)
            partial_scores = evaluator.partial_scores
            if exact_score == 0:
                print("{} pred: {}".format(hardness, p_str))
                print("{} gold: {}".format(hardness, g_str))
                print("")
            scores[hardness]['exact'] += exact_score
            scores['all']['exact'] += exact_score
            for type_ in partial_types:
                if partial_scores[type_]['pred_total'] > 0:
                    scores[hardness]['partial'][type_]['acc'] += partial_scores[type_]['acc']
                    scores[hardness]['partial'][type_]['acc_count'] += 1
                if partial_scores[type_]['label_total'] > 0:
                    scores[hardness]['partial'][type_]['rec'] += partial_scores[type_]['rec']
                    scores[hardness]['partial'][type_]['rec_count'] += 1
                scores[hardness]['partial'][type_]['f1'] += partial_scores[type_]['f1']
                if partial_scores[type_]['pred_total'] > 0:
                    scores['all']['partial'][type_]['acc'] += partial_scores[type_]['acc']
                    scores['all']['partial'][type_]['acc_count'] += 1
                if partial_scores[type_]['label_total'] > 0:
                    scores['all']['partial'][type_]['rec'] += partial_scores[type_]['rec']
                    scores['all']['partial'][type_]['rec_count'] += 1
                scores['all']['partial'][type_]['f1'] += partial_scores[type_]['f1']

            entries.append({
                'predictSQL': p_str,
                'goldSQL': g_str,
                'hardness': hardness,
                'exact': exact_score,
                'partial': partial_scores
            })

    for level in levels:
        if scores[level]['count'] == 0:
            continue
        if etype in ["all", "exec"]:
            scores[level]['exec'] /= scores[level]['count']

        if etype in ["all", "match"]:
            scores[level]['exact'] /= scores[level]['count']
            for type_ in partial_types:
                if scores[level]['partial'][type_]['acc_count'] == 0:
                    scores[level]['partial'][type_]['acc'] = 0
                else:
                    scores[level]['partial'][type_]['acc'] = scores[level]['partial'][type_]['acc'] / \
                        scores[level]['partial'][type_]['acc_count'] * 1.0
                if scores[level]['partial'][type_]['rec_count'] == 0:
                    scores[level]['partial'][type_]['rec'] = 0
                else:
                    scores[level]['partial'][type_]['rec'] = scores[level]['partial'][type_]['rec'] / \
                        scores[level]['partial'][type_]['rec_count'] * 1.0
                if scores[level]['partial'][type_]['acc'] == 0 and scores[level]['partial'][type_]['rec'] == 0:
                    scores[level]['partial'][type_]['f1'] = 1
                else:
                    scores[level]['partial'][type_]['f1'] = \
                        2.0 * scores[level]['partial'][type_]['acc'] * scores[level]['partial'][type_]['rec'] / (
                            scores[level]['partial'][type_]['rec'] + scores[level]['partial'][type_]['acc'])

    print_scores(scores, etype)


def eval_exec_match(db, p_str, g_str, pred, gold):
    """
    return 1 if the values between prediction and gold are matching
    in the corresponding index. Currently not support multiple col_unit(pairs).
    """
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    try:
        cursor.execute(p_str)
        p_res = cursor.fetchall()
    except Exception:
        return False

    cursor.execute(g_str)
    q_res = cursor.fetchall()

    def res_map(res, val_units):
        rmap = {}
        for idx, val_unit in enumerate(val_units):
            key = tuple(val_unit[1]) if not val_unit[2] else (
                val_unit[0], tuple(val_unit[1]), tuple(val_unit[2]))
            rmap[key] = [r[idx] for r in res]
        return rmap

    p_val_units = [unit[1] for unit in pred['select'][1]]
    q_val_units = [unit[1] for unit in gold['select'][1]]
    return res_map(p_res, p_val_units) == res_map(q_res, q_val_units)


# Rebuild SQL functions for value evaluation
def rebuild_cond_unit_val(cond_unit):
    if cond_unit is None or not DISABLE_VALUE:
        return cond_unit

    not_op, op_id, val_unit, val1, val2 = cond_unit
    if type(val1) is not dict:
        val1 = None
    else:
        val1 = rebuild_sql_val(val1)
    if type(val2) is not dict:
        val2 = None
    else:
        val2 = rebuild_sql_val(val2)
    return not_op, op_id, val_unit, val1, val2


def rebuild_condition_val(condition):
    if condition is None or not DISABLE_VALUE:
        return condition

    res = []
    for idx, it in enumerate(condition):
        if idx % 2 == 0:
            res.append(rebuild_cond_unit_val(it))
        else:
            res.append(it)
    return res


def rebuild_sql_val(sql):
    if sql is None or not DISABLE_VALUE:
        return sql

    sql['from']['conds'] = rebuild_condition_val(sql['from']['conds'])
    sql['having'] = rebuild_condition_val(sql['having'])
    sql['where'] = rebuild_condition_val(sql['where'])
    for tab_unit in sql['from']['table_units']:
        if tab_unit[0] == 'sql':
            tab_unit = ('sql', rebuild_sql_val(tab_unit[1]))
    sql['intersect'] = rebuild_sql_val(sql['intersect'])
    sql['except'] = rebuild_sql_val(sql['except'])
    sql['union'] = rebuild_sql_val(sql['union'])

    return sql


# Rebuild SQL functions for foreign key evaluation
def build_valid_col_units(table_units, schema):
    col_names = schema._table['column_names_original']
    tab_ids = [table_unit[1]
               for table_unit in table_units if table_unit[0] == TABLE_TYPE['table_unit']]
    valid_col_units = []
    for value in schema.idMap.values():
        if col_names[value][0] in tab_ids:
            valid_col_units.append(value)
    return valid_col_units


def rebuild_col_unit_col(valid_col_units, col_unit, kmap):
    if col_unit is None:
        return col_unit

    agg_id, col_id, distinct = col_unit
    if col_id in kmap and col_id in valid_col_units:
        col_id = kmap[col_id]
    if DISABLE_DISTINCT:
        distinct = None
    return agg_id, col_id, distinct


def rebuild_val_unit_col(valid_col_units, val_unit, kmap):
    if val_unit is None:
        return val_unit

    unit_op, col_unit1, col_unit2 = val_unit
    col_unit1 = rebuild_col_unit_col(valid_col_units, col_unit1, kmap)
    col_unit2 = rebuild_col_unit_col(valid_col_units, col_unit2, kmap)
    return unit_op, col_unit1, col_unit2


def rebuild_table_unit_col(valid_col_units, table_unit, kmap):
    if table_unit is None:
        return table_unit

    table_type, col_unit_or_sql = table_unit
    if isinstance(col_unit_or_sql, tuple):
        col_unit_or_sql = rebuild_col_unit_col(valid_col_units, col_unit_or_sql, kmap)
    return table_type, col_unit_or_sql


def rebuild_cond_unit_col(valid_col_units, cond_unit, kmap):
    if cond_unit is None:
        return cond_unit

    not_op, op_id, val_unit, val1, val2 = cond_unit
    val_unit = rebuild_val_unit_col(valid_col_units, val_unit, kmap)
    if isinstance(val1, dict):
        val1 = rebuild_sql_col(valid_col_units, val1, kmap)
    if isinstance(val2, dict):
        val2 = rebuild_sql_col(valid_col_units, val2, kmap)
    return not_op, op_id, val_unit, val1, val2


def rebuild_condition_col(valid_col_units, condition, kmap):
    for idx in range(len(condition)):
        if idx % 2 == 0:
            condition[idx] = rebuild_cond_unit_col(valid_col_units, condition[idx], kmap)
    return condition


def rebuild_select_col(valid_col_units, sel, kmap):
    if sel is None:
        return sel
    distinct, _list = sel
    new_list = []
    for it in _list:
        agg_id, val_unit = it
        new_list.append((agg_id, rebuild_val_unit_col(valid_col_units, val_unit, kmap)))
    if DISABLE_DISTINCT:
        distinct = None
    return distinct, new_list


def rebuild_from_col(valid_col_units, from_, kmap):
    if from_ is None:
        return from_

    from_['table_units'] = [rebuild_table_unit_col(valid_col_units, table_unit, kmap) for table_unit in
                            from_['table_units']]
    from_['conds'] = rebuild_condition_col(valid_col_units, from_['conds'], kmap)
    return from_


def rebuild_group_by_col(valid_col_units, group_by, kmap):
    if group_by is None:
        return group_by

    return [rebuild_col_unit_col(valid_col_units, col_unit, kmap) for col_unit in group_by]


def rebuild_order_by_col(valid_col_units, order_by, kmap):
    if order_by is None or len(order_by) == 0:
        return order_by

    direction, val_units = order_by
    new_val_units = [rebuild_val_unit_col(valid_col_units, val_unit, kmap)
                     for val_unit in val_units]
    return direction, new_val_units


def rebuild_sql_col(valid_col_units, sql, kmap):
    if sql is None:
        return sql

    sql['select'] = rebuild_select_col(valid_col_units, sql['select'], kmap)
    sql['from'] = rebuild_from_col(valid_col_units, sql['from'], kmap)
    sql['where'] = rebuild_condition_col(valid_col_units, sql['where'], kmap)
    sql['groupBy'] = rebuild_group_by_col(valid_col_units, sql['groupBy'], kmap)
    sql['orderBy'] = rebuild_order_by_col(valid_col_units, sql['orderBy'], kmap)
    sql['having'] = rebuild_condition_col(valid_col_units, sql['having'], kmap)
    sql['intersect'] = rebuild_sql_col(valid_col_units, sql['intersect'], kmap)
    sql['except'] = rebuild_sql_col(valid_col_units, sql['except'], kmap)
    sql['union'] = rebuild_sql_col(valid_col_units, sql['union'], kmap)

    return sql


def build_foreign_key_map(entry):
    cols_orig = entry["column_names_original"]

    # rebuild cols corresponding to idmap in Schema
    cols = [i for i in range(len(cols_orig))]

    def keyset_in_list(k1, k2, k_list):
        for k_set in k_list:
            if k1 in k_set or k2 in k_set:
                return k_set
        new_k_set = set()
        k_list.append(new_k_set)
        return new_k_set

    foreign_key_list = []
    foreign_keys = entry["foreign_keys"]
    for fkey in foreign_keys:
        key1, key2 = fkey
        key_set = keyset_in_list(key1, key2, foreign_key_list)
        key_set.add(key1)
        key_set.add(key2)

    foreign_key_map = {}
    for key_set in foreign_key_list:
        sorted_list = sorted(list(key_set))
        midx = sorted_list[0]
        for idx in sorted_list:
            foreign_key_map[cols[idx]] = cols[midx]

    return foreign_key_map


def build_foreign_key_map_from_json(table):
    with open(table) as f:
        data = json.load(f)
    tables = {}
    for entry in data:
        tables[entry['db_id']] = build_foreign_key_map(entry)
    return tables


def is_col_valid(col_unit, in_where=False):
    if col_unit is None:
        return True
    if col_unit[0] == 0 and col_unit[1] == 0:
        return False
    if in_where and col_unit[0] != 0:
        return False
    return True


def is_query_valid(_sql, schema):
    select_body = _sql['select'][1]
    if len(select_body) != len(set(select_body)):
        return False

    conds = _sql['from']['conds'][::2]
    tab_in_conds = set()
    for cond in conds:
        tab1 = schema._table['column_names_original'][cond[2][1][1]][0]
        tab2 = schema._table['column_names_original'][cond[3][1]][0]
        if tab1 == -1 or tab2 == -1:
            return False
        tab_in_conds.add(tab1)
        tab_in_conds.add(tab2)

    table_units = _sql['from']['table_units']
    tab_in_from = set()
    for tab in table_units:
        if isinstance(tab[1], int):
            tab_in_from.add(tab[1])

    if len(tab_in_from) > 1 and tab_in_conds != tab_in_from:
        return False

    if len(table_units) == 1 and len(conds) > 0:
        return False

    where_conds = _sql['where'][::2]
    having_conds = _sql['having'][::2]
    for cond in where_conds:
        if isinstance(cond[3], dict) and not is_sql_valid(cond[3], schema):
            return False
        if not is_col_valid(cond[2][1], True):
            return False
        if not is_col_valid(cond[2][2], True):
            return False
    for cond in having_conds:
        if isinstance(cond[3], dict) and not is_sql_valid(cond[3], schema):
            return False
        if not is_col_valid(cond[2][1]):
            return False
        if not is_col_valid(cond[2][2]):
            return False
    groupBy = _sql['groupBy']
    for col_unit in groupBy:
        if not is_col_valid(col_unit):
            return False
    if len(_sql['orderBy']) > 0:
        orderBy = _sql['orderBy'][1]
        for val_unit in orderBy:
            if not is_col_valid(val_unit[1]):
                return False
            if not is_col_valid(val_unit[2]):
                return False
    return True


def is_sql_valid(_sql, schema):
    if _sql['except']:
        if not is_query_valid(_sql['except'], schema):
            return False
    elif _sql['union']:
        if not is_query_valid(_sql['union'], schema):
            return False
    elif _sql['intersect']:
        if not is_query_valid(_sql['intersect'], schema):
            return False

    return is_query_valid(_sql, schema)

# Define new evaluator
@registry.register('evaluator', 'spider')
class SpiderSqlEvaluator:
    def __init__(self, transition_system, args):
        pass

    @staticmethod
    def print_results(results, p_func=print):
        print_scores(results, 'match', p_func=p_func)

    @staticmethod
    def evaluate_dataset(examples, decode_results, out_path, fast_mode=True,
                         test_mode='dev', save_failed_samples=False):
        evaluator = SpiderEvaluator()
        if fast_mode:
            levels = ['easy', 'medium', 'hard', 'extra', 'all']
            partial_types = ['select', 'select(no AGG)', 'where', 'where(no OP)', 'group(no Having)',
                             'group', 'order', 'and/or', 'IUEN', 'keywords']
            etype = 'match'
            scores = {}
            # Init scores
            for level in levels:
                scores[level] = {'count': 0, 'partial': {}, 'exact': 0.}
                for type_ in partial_types:
                    scores[level]['partial'][type_] = {
                        'acc': 0., 'rec': 0., 'f1': 0., 'acc_count': 0, 'rec_count': 0}
            pred_sql = []
            gold_sql = []
            pred_actions = []
            gold_actions = []
            questions = []
            eval_err_num = 0
            idx = 0
            for example, spider_sql in zip(examples, decode_results):
                pruned_hyps = []
                for hyp in spider_sql:
                    if is_sql_valid(hyp.code, example.schema):
                        pruned_hyps.append(hyp)
                gold_spider_sql = example.sql
                _gold_spider_sql = deepcopy(gold_spider_sql)
                pred_spider_sql = pruned_hyps[:1]
                if not pred_spider_sql:
                    # dummy sql
                    surface_sql = 'SELECT *'
                else:
                    surface_sql = SpiderAST(pred_spider_sql[0].code, example.schema._table).get_sql()
                if not pred_spider_sql:
                    p_sql = {
                        "except": None,
                        "from": {
                            "conds": [],
                            "table_units": []
                        },
                        "groupBy": [],
                        "having": [],
                        "intersect": None,
                        "limit": None,
                        "orderBy": [],
                        "select": [
                            False,
                            []
                        ],
                        "union": None,
                        "where": []
                    }
                    pred_spider_sql = SpiderDecodeHypothesis(example.schema)
                    pred_spider_sql.code = p_sql
                    eval_err_num += 1
                else:
                    pred_spider_sql = pred_spider_sql[0]
                _pred_spider_sql = deepcopy(pred_spider_sql.code)

                schema = example.schema
                kmap = build_foreign_key_map(example.schema._table)
                g_valid_col_units = build_valid_col_units(
                    gold_spider_sql['from']['table_units'], schema)
                gold_spider_sql = rebuild_sql_val(gold_spider_sql)
                gold_spider_sql = rebuild_sql_col(g_valid_col_units, gold_spider_sql, kmap)
                p_valid_col_units = build_valid_col_units(
                    pred_spider_sql.code['from']['table_units'], schema)
                pred_spider_sql.code = rebuild_sql_val(pred_spider_sql.code)
                pred_spider_sql.code = rebuild_sql_col(
                    p_valid_col_units, pred_spider_sql.code, kmap)

                hardness = evaluator.eval_hardness(gold_spider_sql)
                scores[hardness]['count'] += 1
                scores['all']['count'] += 1
                exact_score = evaluator.eval_exact_match(pred_spider_sql.code, gold_spider_sql)
                if exact_score == 0 and save_failed_samples:
                    f_out = open(os.path.join(out_path, "%d-%s.md" % (idx, hardness)), "w")
                    f_out.write('### Question\n%s\n' % example.original)
                    f_out.write('\n### Spider SQL\n')
                    f_out.write('- ***pred***: ')
                    f_out.write('%s\n' % surface_sql)
                    f_out.write('- ***gold***: ')
                    f_out.write('%s\n' % example.tgt_code)
                    f_out.write('\n### Action Sequences Diff\n')
                    pred_actions = []
                    for a in pred_spider_sql.actions:
                        pred_actions.append(str(a).replace('*', '\*'))
                    gold_actions = []
                    for a in example.tgt_actions:
                        gold_actions.append(str(a.action).replace('*', '\*'))
                    for line in difflib.unified_diff(pred_actions, gold_actions, fromfile='pred', tofile='gold'):
                        f_out.write('\t%s\n' % line)

                    f_out.write('\n### Schema\n')
                    f_out.write('\tcol_id,\ttab_name,\tcol_name\n')
                    for _id, (tab_id, col_name) in enumerate(example.schema._table['exp_column_names_original']):
                        f_out.write('\t%d,\t%s,\t%s\n' % (
                            _id, example.schema._table['exp_table_names_original'][tab_id], col_name))
                    f_out.write('\n### Primary Keys\n%s\n' %
                                str(example.schema._table['exp_primary_keys']))
                    f_out.close()
                    questions.append(" ".join(example.src_sent))
                    pred_sql.append(_pred_spider_sql)
                    gold_sql.append(_gold_spider_sql)
                    pred_actions.append(pred_spider_sql.actions)
                    gold_actions.append([a.action for a in example.tgt_actions])
                partial_scores = evaluator.partial_scores
                scores[hardness]['exact'] += exact_score
                scores['all']['exact'] += exact_score
                for type_ in partial_types:
                    if partial_scores[type_]['pred_total'] > 0:
                        scores[hardness]['partial'][type_]['acc'] += partial_scores[type_]['acc']
                        scores[hardness]['partial'][type_]['acc_count'] += 1
                    if partial_scores[type_]['label_total'] > 0:
                        scores[hardness]['partial'][type_]['rec'] += partial_scores[type_]['rec']
                        scores[hardness]['partial'][type_]['rec_count'] += 1
                    scores[hardness]['partial'][type_]['f1'] += partial_scores[type_]['f1']
                    if partial_scores[type_]['pred_total'] > 0:
                        scores['all']['partial'][type_]['acc'] += partial_scores[type_]['acc']
                        scores['all']['partial'][type_]['acc_count'] += 1
                    if partial_scores[type_]['label_total'] > 0:
                        scores['all']['partial'][type_]['rec'] += partial_scores[type_]['rec']
                        scores['all']['partial'][type_]['rec_count'] += 1
                    scores['all']['partial'][type_]['f1'] += partial_scores[type_]['f1']
                idx += 1

            for level in levels:
                if scores[level]['count'] == 0:
                    continue

                if etype in ["all", "match"]:
                    scores[level]['exact'] /= scores[level]['count']
                    for type_ in partial_types:
                        if scores[level]['partial'][type_]['acc_count'] == 0:
                            scores[level]['partial'][type_]['acc'] = 0
                        else:
                            scores[level]['partial'][type_]['acc'] = scores[level]['partial'][type_]['acc'] /\
                                scores[level]['partial'][type_]['acc_count'] * 1.0
                        if scores[level]['partial'][type_]['rec_count'] == 0:
                            scores[level]['partial'][type_]['rec'] = 0
                        else:
                            scores[level]['partial'][type_]['rec'] = scores[level]['partial'][type_]['rec'] /\
                                scores[level]['partial'][type_]['rec_count'] * 1.0
                        if scores[level]['partial'][type_]['acc'] == 0 and scores[level]['partial'][type_]['rec'] == 0:
                            scores[level]['partial'][type_]['f1'] = 1
                        else:
                            scores[level]['partial'][type_]['f1'] = \
                                2.0 * scores[level]['partial'][type_]['acc'] * scores[level]['partial'][type_]['rec'] /\
                                (scores[level]['partial'][type_]['rec'] + scores[level]['partial'][type_]['acc'])

            scores["accuracy"] = scores["all"]["exact"]

            out_dict = {
                "questions": questions,
                "pred_sql": pred_sql,
                "gold_sql": gold_sql,
                "pred_actions": pred_actions,
                "gold_actions": gold_actions,
            }
            print("eval_err_num:{}".format(eval_err_num))
            if save_failed_samples:
                with open(os.path.join(out_path, "failed_samples.pkl"), "wb") as out:
                    pickle.dump(out_dict, out)
        else:
            scores = {'accuracy': 0.0}
            for example, spider_sql in zip(examples, decode_results):
                pruned_hyps = []
                for hyp in spider_sql:
                    if is_sql_valid(hyp.code, example.schema):
                        pruned_hyps.append(hyp)
                gold_spider_sql = example.sql
                schema = example.schema
                kmap = build_foreign_key_map(example.schema._table)
                g_valid_col_units = build_valid_col_units(
                    gold_spider_sql['from']['table_units'], schema)
                gold_spider_sql = rebuild_sql_val(gold_spider_sql)
                gold_spider_sql = rebuild_sql_col(g_valid_col_units, gold_spider_sql, kmap)

                flag = False
                for hyp in pruned_hyps:
                    p_valid_col_units = build_valid_col_units(
                        hyp.code['from']['table_units'], schema)
                    hyp.code = rebuild_sql_val(hyp.code)
                    hyp.code = rebuild_sql_col(p_valid_col_units, hyp.code, kmap)
                    exact_score = evaluator.eval_exact_match(hyp.code, gold_spider_sql)
                    if exact_score:
                        flag = True
                        break
                scores['accuracy'] += 1.0 if flag else 0.0
            scores['accuracy'] /= len(examples)

        return scores


@registry.register('evaluator', 'spider-action-evaluator')
def create_spider_action_prediction_evaluator(transition_system, eval_top_pred_only=True, for_inference=False):
    def evaluate_action_predictions(examples, predictions, exp_dir_path):
        """
        @param examples_batches: list(list(example))
        @param predictions_batches: list(list(tensor))
        @param exp_dir_path: str
        @return:
        """
        eva_output_path = None
        if (exp_dir_path is not None) and (not for_inference):
            eva_output_path = os.path.join(exp_dir_path, 'eval_output')
            if not os.path.exists(eva_output_path):
                os.makedirs(eva_output_path)
        logger = logging.getLogger()

        code_results = list()
        for data_idx, pred in enumerate(predictions):
            decoded_hyps = list()
            for hyp_id, hyp in enumerate(pred):
                try:
                    hyp.code = transition_system.ast_to_surface_code(
                        hyp.tree, examples[data_idx].schema
                    )
                    decoded_hyps.append(hyp)
                except Exception:
                    logger.error('Exception in converting tree to code:')
                    logger.error(traceback.format_stack())
                    logger.error(traceback.format_exc())
                    logger.error('-' * 60)
                    logger.error('Example: %s\nIntent: %s\nTarget Code:\n%s\nHypothesis[%d]:\n%s' % (
                        data_idx, ' '.join(examples[data_idx].src_sent),
                        examples[data_idx].tgt_code, hyp_id, hyp.tree.to_string()
                    ))
                    logger.error('-' * 60)
            code_results.append(decoded_hyps)

        if for_inference:
            eval_result = []
            for example, spider_sql in zip(examples, code_results):
                pruned_hyps = []
                for hyp in spider_sql:
                    if is_sql_valid(hyp.code, example.schema):
                        pruned_hyps.append(hyp)
                pred_spider_sql = pruned_hyps[:1]
                if not pred_spider_sql:
                    # dummy sql
                    surface_sql = 'SELECT *'
                else:
                    pred_spider_sql = pred_spider_sql[0].code
                    surface_sql = SpiderAST(pred_spider_sql, example.schema._table).get_sql()
                eval_result.append(surface_sql)
            with open('predicted_sql.txt', 'w') as f:
                for q in eval_result:
                    f.write(q + '\n')
        else:
            evaluator = SpiderSqlEvaluator(None, None)
            eval_results = evaluator.evaluate_dataset(
                examples, code_results, eva_output_path, fast_mode=eval_top_pred_only,
                test_mode='dev', save_failed_samples=eva_output_path is not None)
            print_scores(eval_results, 'match', p_func=logger.info)
            eval_result = eval_results['accuracy']
        return eval_result

    return evaluate_action_predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold', dest='gold', type=str)
    parser.add_argument('--pred', dest='pred', type=str)
    parser.add_argument('--db', dest='db', type=str)
    parser.add_argument('--table', dest='table', type=str)
    parser.add_argument('--etype', dest='etype', type=str)
    args = parser.parse_args()

    gold = args.gold
    pred = args.pred
    db_dir = args.db
    table = args.table
    etype = args.etype

    assert etype in ["all", "exec", "match"], "Unknown evaluation method"

    kmaps = build_foreign_key_map_from_json(table)

    evaluate(gold, pred, db_dir, etype, kmaps)
