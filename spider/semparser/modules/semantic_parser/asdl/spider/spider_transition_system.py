# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from copy import deepcopy
from collections import defaultdict

from semparser.common import registry
from semparser.modules.semantic_parser.asdl.asdl_ast import RealizedField, AbstractSyntaxTree
from semparser.modules.semantic_parser.asdl.transition_system import GenTokenAction, TransitionSystem, \
    ApplyRuleAction, ReduceAction

WHERE_OPS = ('Not', 'Between', 'Equal', 'Greater', 'Less', 'GreaterEqual', 'LessEqual',
             'Inequal', 'In', 'Like', 'Is', 'Exists')
UNIT_OPS = ('none', 'minus', 'plus', "multiply", 'divide')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
TABLE_TYPE = {
    'sql': "sql",
    'table_unit': "table_unit",
}

COND_OPS = ('and', 'or')
SQL_OPS = ('intersect', 'union', 'except')
ORDER_OPS = ('desc', 'asc')

NUMBER = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten')

VAL_PLACEHOLDER = '"value"'
NUM_PLACEHOLDER = 1

class SpiderGenTokenAction(GenTokenAction):
    def __init__(self, token):
        super(SpiderGenTokenAction, self).__init__(token)

    def __eq__(self, other):
        return isinstance(other, SpiderGenTokenAction) and self.token == other.token

    def is_stop_signal(self):
        return True

class SpiderSelectColumnAction(GenTokenAction):
    def __init__(self, column_id):
        super(SpiderSelectColumnAction, self).__init__(column_id)

    def __eq__(self, other):
        return isinstance(other, SpiderSelectColumnAction) and self.token == other.token

    @property
    def column_id(self):
        return self.token

    def __repr__(self):
        return 'SelectColumnAction[id=%d]' % self.column_id

class SpiderSelectTableAction(GenTokenAction):
    def __init__(self, table_id):
        super(SpiderSelectTableAction, self).__init__(table_id)

    def __eq__(self, other):
        return isinstance(other, SpiderSelectTableAction) and self.token == other.token

    @property
    def table_id(self):
        return self.token

    def __repr__(self):
        return 'SelectTableAction[id=%d]' % self.table_id


@registry.register('transition_system', 'spider')
class SpiderTransitionSystem(TransitionSystem):
    def __init__(self, grammar):
        super(SpiderTransitionSystem, self).__init__(grammar)

    def ast_to_surface_code(self, asdl_ast, schema):
        return asdl_ast_to_sql_query(asdl_ast, schema)

    def compare_ast(self, hyp_ast, ref_ast):
        raise NotImplementedError

    def tokenize_code(self, code, mode):
        raise NotImplementedError

    def surface_code_to_ast(self, code):
        raise NotImplementedError

    def get_valid_continuation_types(self, hyp):
        if hyp.tree:
            if self.grammar.is_composite_type(hyp.frontier_field.type):
                if hyp.frontier_field.cardinality == 'single':
                    return ApplyRuleAction,
                else:  # optional, multiple
                    return ApplyRuleAction, ReduceAction
            elif hyp.frontier_field.type.name == 'col_id':
                if hyp.frontier_field.cardinality == 'single':
                    return SpiderSelectColumnAction,
                elif hyp.frontier_field.cardinality == 'optional':
                    return SpiderSelectColumnAction, ReduceAction
            elif hyp.frontier_field.type.name == 'tab_id':
                if hyp.frontier_field.cardinality == 'single':
                    return SpiderSelectTableAction,
                elif hyp.frontier_field.cardinality == 'optional':
                    return SpiderSelectTableAction, ReduceAction
            else:
                if hyp.frontier_field.cardinality == 'single':
                    return SpiderGenTokenAction,
                else:
                    return SpiderGenTokenAction, ReduceAction
        else:
            return ApplyRuleAction,

    def get_primitive_field_actions(self, realized_field):
        if realized_field.type.name == 'number':
            return [SpiderGenTokenAction('</primitive>')]
        elif realized_field.type.name == 'string':
            return [SpiderGenTokenAction('</primitive>')]
        elif realized_field.type.name == 'col_id':
            return [SpiderSelectColumnAction(int(realized_field.value))]
        elif realized_field.type.name == 'tab_id':
            return [SpiderSelectTableAction(int(realized_field.value))]
        else:
            raise ValueError('unknown primitive field type')


def asdl_ast_to_sql_query(ast, schema):
    def get_ast_prod_name(ast):
        return ast.production.constructor.name

    def reconstr_split(_ast, split_constr_name):
        split_constr_name = split_constr_name[0].lower() + split_constr_name[1:]
        _sql = {'union': None, 'intersect': None, 'except': None}
        l_ast = _ast.fields[0].value
        r_ast = _ast.fields[1].value
        _sql.update(reconstr_single_sql(l_ast))
        _sql[split_constr_name] = reconstr_single_sql(r_ast)
        return _sql

    def reconstr_agg_op_sql(agg_op_ast):
        return AGG_OPS.index(get_ast_prod_name(agg_op_ast))

    def reconstr_unit_op_sql(unit_op_ast):
        if unit_op_ast is None:
            return 0
        else:
            return UNIT_OPS.index(get_ast_prod_name(unit_op_ast))

    def reconstr_col_id_sql(col_id_ast):
        return schema._table['col_exp_to_ori'][col_id_ast]

    def reconstr_col_unit_sql(col_unit_ast):
        assert get_ast_prod_name(col_unit_ast) == 'ColUnit'
        col_unit_val_list = []
        # agg_op?
        agg_op_ast = col_unit_ast.fields[0].value
        col_unit_val_list.append(reconstr_agg_op_sql(agg_op_ast))
        # col_id
        col_id_ast = col_unit_ast.fields[1].value
        col_unit_val_list.append(reconstr_col_id_sql(col_id_ast))
        # is_distinct
        if len(col_unit_ast.fields) == 3:
            is_distinct_ast = col_unit_ast.fields[2].value
            is_distinct = True if get_ast_prod_name(is_distinct_ast) == 'true' else False
            col_unit_val_list.append(is_distinct)
        else:
            col_unit_val_list.append(False)
        return tuple(col_unit_val_list)

    def reconstr_val_unit_sql(val_unit_ast):
        prod_name = get_ast_prod_name(val_unit_ast)
        assert 'ValUnit' in prod_name
        val_unit_val_list = []
        # unit_op?
        unit_op_name = prod_name[7:].lower()
        if unit_op_name == "":
            unit_op_name = 'none'
        unit_op = UNIT_OPS.index(unit_op_name)
        val_unit_val_list.append(unit_op)
        # col_unit
        col_unit_1_ast = val_unit_ast.fields[0].value
        val_unit_val_list.append(reconstr_col_unit_sql(col_unit_1_ast))
        # col_unit?
        if unit_op == 0:
            val_unit_val_list.append(None)
        else:
            col_unit_2_ast = val_unit_ast.fields[1].value
            val_unit_val_list.append(reconstr_col_unit_sql(col_unit_2_ast))
        return tuple(val_unit_val_list)

    def reconstr_select_col_sql(sel_col_ast):
        assert get_ast_prod_name(sel_col_ast) == 'SelectCol'
        sel_col_val_list = []
        # agg_op?
        agg_op_ast = sel_col_ast.fields[0].value
        sel_col_val_list.append(reconstr_agg_op_sql(agg_op_ast))
        # val_unit
        val_unit_ast = sel_col_ast.fields[1].value
        sel_col_val_list.append(reconstr_val_unit_sql(val_unit_ast))
        return tuple(sel_col_val_list)

    def reconstr_select_sql(select_ast):
        assert get_ast_prod_name(select_ast) == 'Select'
        select_val_list = [False]
        # sel_col
        sel_col_sql_list = []
        for sel_col_ast in select_ast.fields[0].value:
            sel_col_sql_list.append(reconstr_select_col_sql(sel_col_ast))
        select_val_list.append(sel_col_sql_list)
        return tuple(select_val_list)

    def reconstr_table_unit_sql(table_unit_ast):
        if get_ast_prod_name(table_unit_ast) == 'ColIdTableUnit':
            col_id_ast = table_unit_ast.fields[0].value
            return 'table_unit', reconstr_col_id_sql(col_id_ast)
        elif get_ast_prod_name(table_unit_ast) == 'QueryTableUnit':
            query_ast = table_unit_ast.fields[0].value
            return 'sql', asdl_ast_to_sql_query(query_ast)
        else:
            raise ValueError

    def reconstr_not_op_sql(not_op_ast):
        if get_ast_prod_name(not_op_ast) == 'True':
            return True
        elif get_ast_prod_name(not_op_ast) == 'False':
            return False
        else:
            raise ValueError

    def reconstr_op_id_sql(op_id_ast):
        return WHERE_OPS.index(get_ast_prod_name(op_id_ast))

    def reconstr_num_id_sql(num_id_ast):
        return NUMBER.index(get_ast_prod_name(num_id_ast))

    def reconstr_val_sql(val_ast):
        val_prod_name = get_ast_prod_name(val_ast)
        if val_prod_name == 'NumVal':
            number = val_ast.fields[0].value
            return number
        elif val_prod_name == 'StrVal':
            string = val_ast.fields[0].value
            return string
        elif val_prod_name == 'QueryVal':
            query_ast = val_ast.fields[0].value
            query_sql = asdl_ast_to_sql_query(query_ast, schema)
            return query_sql
        elif val_prod_name == 'ColUnitVal':
            col_unit_ast = val_ast.fields[0].value
            col_unit_sql = reconstr_col_unit_sql(col_unit_ast)
            return col_unit_sql
        elif val_prod_name == 'ListVal':
            num_id_ast = val_ast.fields[0].value
            num_id = reconstr_num_id_sql(num_id_ast)
            return [VAL_PLACEHOLDER] * num_id
        else:
            raise ValueError

    def reconstr_cond_unit_sql(cond_unit_ast):
        cond_name = get_ast_prod_name(cond_unit_ast)
        cond_unit_val_list = []

        not_op = False
        if cond_name.startswith('Not'):
            not_op = True
            cond_name = cond_name[3:]
        cond_unit_val_list.append(not_op)
        if_nested = False
        if cond_name.startswith('Nested') or cond_name in ['In', 'NotIn']:
            if_nested = True
            if cond_name.startswith('Nested'):
                cond_name = cond_name[6:]

        assert cond_name in WHERE_OPS, "unknown condition name: %s" % cond_name

        op_id = WHERE_OPS.index(cond_name)
        cond_unit_val_list.append(op_id)

        val_unit_ast = cond_unit_ast.fields[0].value
        val_unit_sql = reconstr_val_unit_sql(val_unit_ast)
        cond_unit_val_list.append(val_unit_sql)
        if if_nested:
            val_ast = cond_unit_ast.fields[1].value
            val_sql = reconstr_single_sql(val_ast)
            cond_unit_val_list.append(val_sql)
        else:
            cond_unit_val_list.append(VAL_PLACEHOLDER)

        if op_id == 1:
            cond_unit_val_list.append(VAL_PLACEHOLDER)
        else:
            cond_unit_val_list.append(None)
        return tuple(cond_unit_val_list)

    def reconstr_cond_unit_conj_sql(cond_unit_conj_ast):
        assert get_ast_prod_name(cond_unit_conj_ast) == 'ConditionUnitConj'
        cond_unit_conj_val_list = []
        # cond_unit
        cond_unit_ast = cond_unit_conj_ast.fields[0].value
        cond_uint_sql = reconstr_cond_unit_sql(cond_unit_ast)
        cond_unit_conj_val_list.append(cond_uint_sql)
        # conj_op?
        conj_op_ast = cond_unit_conj_ast.fields[1].value
        if conj_op_ast:
            cond_unit_conj_val_list.append(get_ast_prod_name(conj_op_ast))
        return cond_unit_conj_val_list

    def reconstr_cond_sql(cond_ast):
        cond_name = get_ast_prod_name(cond_ast)
        if cond_name in ["And", "Or"]:
            cond_list1 = reconstr_cond_sql(cond_ast.fields[0].value)
            cond_list2 = reconstr_cond_sql(cond_ast.fields[1].value)
            return cond_list1 + [cond_name.lower()] + cond_list2
        else:
            return [reconstr_cond_unit_sql(cond_ast)]

    def reconstr_from_sql(selected_tabs, select_ast, from_ast):
        from_val_list = {}

        table_unit_sql_list = []
        for tab in selected_tabs:
            if tab == -1:
                continue
            table_unit_sql_list.append(('table_unit', tab))
        cond_list = []
        if from_ast is not None:
            for cond in from_ast.fields[0].value:
                tab1 = cond.fields[0].value.additional_info
                tab2 = cond.fields[1].value.additional_info
                if tab1 == tab2:
                    table_unit_sql_list.append(('table_unit', tab1))
                col1 = reconstr_col_unit_sql(cond.fields[0].value)
                col2 = reconstr_col_unit_sql(cond.fields[1].value)
                cond_list.append((False, 2, (0, col1, None), col2, None))
                cond_list.append('and')
            cond_list = cond_list[:-1]
        from_val_list['table_units'] = table_unit_sql_list
        from_val_list['conds'] = cond_list
        return from_val_list

    def reconstr_where_sql(where_ast):
        assert get_ast_prod_name(where_ast) == 'Where'
        cond_ast = where_ast.fields[0].value
        cond_sql = reconstr_cond_sql(cond_ast)
        return cond_sql

    def reconstr_having_sql(having_ast):
        assert get_ast_prod_name(having_ast) == 'Having'
        cond_ast = having_ast.fields[0].value
        cond_sql = reconstr_cond_sql(cond_ast)
        return cond_sql

    def reconstr_order_op_sql(order_op_ast):
        return get_ast_prod_name(order_op_ast)

    def reconstr_order_by_sql(order_by_ast):
        prod_name = get_ast_prod_name(order_by_ast)
        assert prod_name in ['Order', 'Superlative']
        order_by_val_list = []
        # order_op
        order_op_ast = order_by_ast.fields[0].value
        order_op_sql = reconstr_order_op_sql(order_op_ast)
        order_by_val_list.append(order_op_sql)
        # val_unit*
        val_unit_sql_list = []
        for val_unit_ast in order_by_ast.fields[1].value:
            val_unit_sql_list.append(reconstr_val_unit_sql(val_unit_ast))
        order_by_val_list.append(val_unit_sql_list)
        return tuple(order_by_val_list), prod_name == 'Superlative'

    def reconstr_group_by_sql(group_by_ast):
        assert get_ast_prod_name(group_by_ast) == 'GroupBy'
        group_by_sql_list = []
        for col_unit in group_by_ast.fields[0].value:
            group_by_sql_list.append(reconstr_col_unit_sql(col_unit))
        return group_by_sql_list

    def find_path(x, y, graph):
        nodes_cnt = len(schema._table['table_names'])
        marked = [-1] * nodes_cnt
        cur = x
        marked[cur] = cur
        idx = 0
        q = [x]
        while idx < len(q):
            cur = q[idx]
            idx += 1
            for t in graph[cur]:
                if marked[t] < 0:
                    q.append(t)
                    marked[t] = cur
                    if t == y:
                        path = []
                        tmp = y
                        while tmp != x:
                            path.append(tmp)
                            tmp = marked[tmp]
                        return path
        return [x, y]

    def expand_tabs(tabs):
        graph = defaultdict(list)
        for x, y in schema._table['foreign_keys']:
            x = schema._table['column_names_original'][x][0]
            y = schema._table['column_names_original'][y][0]
            graph[x].append(y)
            graph[y].append(x)
        tabs = list(tabs)
        tab_cnt = len(tabs)
        expanded_tabs = set(tabs)
        for i in range(tab_cnt):
            for j in range(i + 1, tab_cnt):
                x = tabs[i]
                y = tabs[j]
                if x == -1 or y == -1:
                    continue
                path = find_path(x, y, graph)
                for t in path:
                    expanded_tabs.add(t)
        return expanded_tabs

    def infer_selected_tabs(query_ast):
        tabs = set()
        select_ast = query_ast.fields[0].value
        for sel_col_ast in select_ast.fields[0].value:
            tmp = get_tabs_from_val_unit(sel_col_ast.fields[1].value)
            for t in tmp:
                tabs.add(t)
        from_ast = query_ast.fields[1].value
        if from_ast is not None and get_ast_prod_name(from_ast) == 'From':
            for cond in from_ast.fields[0].value:
                tabs.add(cond.fields[0].value.additional_info)
                tabs.add(cond.fields[1].value.additional_info)
        where_ast = query_ast.fields[2].value
        if where_ast is not None:
            for t in get_tabs_from_cond(where_ast.fields[0].value):
                tabs.add(t)
        having_ast = query_ast.fields[3].value
        if having_ast is not None:
            for t in get_tabs_from_cond(having_ast.fields[0].value):
                tabs.add(t)
        order_by_ast = query_ast.fields[4].value
        if order_by_ast is not None:
            for val_unit in order_by_ast.fields[1].value:
                tmp = get_tabs_from_val_unit(val_unit)
                for t in tmp:
                    tabs.add(t)
        group_by_ast = query_ast.fields[5].value
        if group_by_ast is not None:
            for col_unit in group_by_ast.fields[0].value:
                tabs.add(col_unit.additional_info)
        return tabs

    def get_tabs_from_val_unit(val_unit_ast):
        prod_name = get_ast_prod_name(val_unit_ast)
        assert 'ValUnit' in prod_name
        tabs = []
        # unit_op?
        unit_op_name = prod_name[7:].lower()
        if unit_op_name == "":
            unit_op_name = 'none'
        unit_op = UNIT_OPS.index(unit_op_name)
        # col_unit
        col_unit_1_ast = val_unit_ast.fields[0].value
        tabs.append(col_unit_1_ast.additional_info)
        # col_unit?
        if unit_op != 0:
            col_unit_2_ast = val_unit_ast.fields[1].value
            tmp = col_unit_2_ast.additional_info
            tabs.append(tmp)
        return tabs

    def get_tabs_from_cond(cond_ast):
        cond_name = get_ast_prod_name(cond_ast)
        if cond_name in ["And", "Or"]:
            tab_list1 = get_tabs_from_cond(cond_ast.fields[0].value)
            tab_list2 = get_tabs_from_cond(cond_ast.fields[1].value)
            return tab_list1 + tab_list2
        else:
            res = get_tabs_from_val_unit(cond_ast.fields[0].value)
            return res

    def reconstr_single_sql(_ast):
        _sql = {'union': None, 'intersect': None, 'except': None}
        query_ast = _ast
        assert get_ast_prod_name(query_ast) == 'Query', get_ast_prod_name(query_ast)
        selected_tabs = infer_selected_tabs(query_ast)
        # sel_expr
        select_ast = query_ast.fields[0].value
        select_sql = reconstr_select_sql(select_ast)
        _sql['select'] = select_sql
        # from_expr
        from_ast = query_ast.fields[1].value
        from_sql = reconstr_from_sql(selected_tabs, select_ast, from_ast)
        _sql['from'] = from_sql
        # where_expr?
        where_ast = query_ast.fields[2].value
        if where_ast is None:
            where_sql = []
        else:
            where_sql = reconstr_where_sql(where_ast)
        _sql['where'] = where_sql
        # having_expr?
        having_ast = query_ast.fields[3].value
        if having_ast is None:
            having_sql = []
        else:
            having_sql = reconstr_having_sql(having_ast)
        _sql['having'] = having_sql
        # orderBy_expr?
        order_by_ast = query_ast.fields[4].value
        if order_by_ast is None:
            order_by_sql, if_limit = [], False
        else:
            order_by_sql, if_limit = reconstr_order_by_sql(order_by_ast)
        _sql['orderBy'] = order_by_sql
        # groupBy_expr?
        group_by_ast = query_ast.fields[5].value
        if group_by_ast is None:
            group_by_sql = []
        else:
            group_by_sql = reconstr_group_by_sql(group_by_ast)
        _sql['groupBy'] = group_by_sql
        # limit_expr?
        if not if_limit:
            limit_sql = None
        else:
            limit_sql = NUM_PLACEHOLDER
        _sql['limit'] = limit_sql
        return _sql

    prod_name = get_ast_prod_name(ast)
    if prod_name in ['Union', 'Intersect', 'Except']:
        sql = reconstr_split(ast, prod_name)
    else:
        assert prod_name == 'SingleQuery', prod_name
        sql = reconstr_single_sql(ast.fields[0].value)
    return sql


def sql_query_to_asdl_ast(sql, grammar, schema):

    def split_query_by_keywords(_sql, keyword):
        corresponding_grammar_keyword = keyword[0].upper() + keyword[1:]
        split_production = grammar.get_prod_by_ctr_name(corresponding_grammar_keyword)
        r_sub_sql = _sql.pop(keyword)
        _sql[keyword] = None
        l_sub_sql = deepcopy(_sql)
        fields = [RealizedField(split_production['lbody'], prod_query_expr(l_sub_sql)),
                  RealizedField(split_production['rbody'], prod_query_expr(r_sub_sql))]
        return AbstractSyntaxTree(split_production, fields)

    def prod_agg_op_ast(agg_op_val):
        agg_op_name = AGG_OPS[agg_op_val]
        agg_op_prod = grammar.get_prod_by_ctr_name(agg_op_name)
        return AbstractSyntaxTree(agg_op_prod)

    def prod_not_op_ast(not_op_val):
        if not_op_val:
            not_op_prod = grammar.get_prod_by_ctr_name('True')
        else:
            assert not_op_val is False
            not_op_prod = grammar.get_prod_by_ctr_name('False')
        return AbstractSyntaxTree(not_op_prod)

    def prod_op_id_ast(op_id_val):
        op_id = WHERE_OPS[op_id_val]
        op_id_prod = grammar.get_prod_by_ctr_name(op_id)
        return AbstractSyntaxTree(op_id_prod)

    def prod_num_ast(num_val):
        num_id = NUMBER[num_val]
        num_id_prod = grammar.get_prod_by_ctr_name(num_id)
        return AbstractSyntaxTree(num_id_prod)

    def prod_order_op_ast(order_op_val):
        op_val_prod = grammar.get_prod_by_ctr_name(order_op_val)
        return AbstractSyntaxTree(op_val_prod)

    def prod_conj_op_ast(conj_op_val):
        conj_op_prod = grammar.get_prod_by_ctr_name(conj_op_val)
        return AbstractSyntaxTree(conj_op_prod)

    def prod_select_col_ast(sel_col_val, tabs):
        select_col_prod = grammar.get_prod_by_ctr_name('SelectCol')
        fileds = []
        agg_op_val_field = RealizedField(select_col_prod['agg_op_val'], prod_agg_op_ast(sel_col_val[0]))
        fileds.append(agg_op_val_field)
        val_unit_val_field = RealizedField(select_col_prod['val_unit_val'], prod_val_unit_ast(sel_col_val[1], tabs))
        fileds.append(val_unit_val_field)
        return AbstractSyntaxTree(select_col_prod, fileds)

    def prod_select_tab_ast(from_body, tabs):
        table_units = from_body['table_units']
        conds = from_body['conds']

        prod_name = 'SelectTab'
        tab_num = min(len(table_units), 4)
        nested = False
        if tab_num == 1 and table_units[0][0] == 'sql':
            nested = True
        if nested:
            prod_name = 'Nested' + prod_name
        else:
            prod_name = prod_name + NUMBER[tab_num - 1]
        select_tab_prod = grammar.get_prod_by_ctr_name(prod_name)

        fields = []
        if nested:
            body_ast = sql_query_to_asdl_ast(table_units[0][1], grammar)
            body_field = RealizedField(select_tab_prod['body'], body_ast)
            fields.append(body_field)
        else:
            for i in range(tab_num):
                table_unit = table_units[i]
                tab_id_val_field = RealizedField(
                    select_tab_prod['tab_id_val%d' % (i + 1)], table_unit[1])
                fields.append(tab_id_val_field)

        cond_val_from_ast = None
        if conds:
            cond_val_from_ast = prod_cond_ast(from_body['conds'], tabs)

        select_tab_ast = AbstractSyntaxTree(select_tab_prod, fields, cond_val_from_ast)

        return select_tab_ast

    def prod_val_ast(val):
        if val is None:
            return None
        if isinstance(val, float):
            val_ast = None
        elif isinstance(val, str):
            val_ast = None
        elif isinstance(val, dict):
            # prod QueryVal here  -- composite
            query_val_prod = grammar.get_prod_by_ctr_name('QueryVal')
            query_val_ast = sql_query_to_asdl_ast(val, grammar, schema)
            query_val_field = RealizedField(query_val_prod['body'], query_val_ast)
            val_ast = AbstractSyntaxTree(query_val_prod, [query_val_field])
        elif isinstance(val, tuple):  # col_unit  -- composite
            col_unit_val_prod = grammar.get_prod_by_ctr_name('ColUnitVal')
            col_unit_val_ast = prod_col_unit_ast(val, [])
            col_unit_val_field = RealizedField(col_unit_val_prod['col_unit_val'], col_unit_val_ast)
            val_ast = AbstractSyntaxTree(col_unit_val_prod, [col_unit_val_field])
        elif isinstance(val, list):
            list_val_prod = grammar.get_prod_by_ctr_name('ListVal')
            list_val_ast = prod_num_ast(len(val))
            list_val_field = RealizedField(list_val_prod['num_val'], list_val_ast)
            val_ast = AbstractSyntaxTree(list_val_prod, [list_val_field])
        else:
            raise ValueError
        return val_ast

    def prod_cond_unit_ast(cond_unit_val, tabs):
        cond_unit_prod = grammar.get_prod_by_ctr_name('CondUnit')
        not_op_val, op_id_val, val_unit_val, val_1, val_2 = cond_unit_val
        cond_unit_val_field_list = []
        # not_op_val
        not_op_val_ast = prod_not_op_ast(not_op_val)
        not_op_val_field = RealizedField(cond_unit_prod['not_op_val'], not_op_val_ast)
        cond_unit_val_field_list.append(not_op_val_field)
        # op_id_val
        op_id_val_ast = prod_op_id_ast(op_id_val)
        op_id_val_field = RealizedField(cond_unit_prod['op_id_val'], op_id_val_ast)
        cond_unit_val_field_list.append(op_id_val_field)
        # val_unit_val
        val_unit_val_ast = prod_val_unit_ast(val_unit_val, tabs)
        val_unit_val_field = RealizedField(cond_unit_prod['val_unit_val'], val_unit_val_ast)
        cond_unit_val_field_list.append(val_unit_val_field)
        # val_1
        val_1_ast = prod_val_ast(val_1)
        val_1_field = RealizedField(cond_unit_prod['val_1'], val_1_ast)
        cond_unit_val_field_list.append(val_1_field)
        if val_2:
            val_2_ast = prod_val_ast(val_2)
            val_2_field = RealizedField(cond_unit_prod['val_2'], val_2_ast)
        else:
            val_2_field = RealizedField(cond_unit_prod['val_2'])
        cond_unit_val_field_list.append(val_2_field)
        return AbstractSyntaxTree(cond_unit_prod, cond_unit_val_field_list)

    def prod_cond_unit_conj_ast(cond_unit_conj_val):
        cond_unit_conj_prod = grammar.get_prod_by_ctr_name('ConditionUnitConj')
        cond_unit_val, conj_op_val = cond_unit_conj_val
        cond_unit_conj_val_field_list = []
        cond_unit_val_ast = prod_cond_unit_ast(cond_unit_val)
        cond_unit_val_field = RealizedField(cond_unit_conj_prod['cond_unit_val'], cond_unit_val_ast)
        cond_unit_conj_val_field_list.append(cond_unit_val_field)
        if conj_op_val != 'none':
            conj_op_val_ast = prod_conj_op_ast(conj_op_val)
            conj_op_val_field = RealizedField(cond_unit_conj_prod['conj_op_val'], conj_op_val_ast)
        else:
            conj_op_val_field = RealizedField(cond_unit_conj_prod['conj_op_val'])
        cond_unit_conj_val_field_list.append(conj_op_val_field)
        return AbstractSyntaxTree(cond_unit_conj_prod, cond_unit_conj_val_field_list)

    def prod_cond_ast(cond_val, tabs):
        if len(cond_val) == 1:
            not_op_val, op_id_val, val_unit_val, val_1, val_2 = cond_val[0]
            cond_name = WHERE_OPS[op_id_val]
            if not_op_val:
                cond_name = 'Not' + cond_name
            if isinstance(val_1, dict) and op_id_val not in [1, 8]:
                cond_name = 'Nested' + cond_name
            cond_prod = grammar.get_prod_by_ctr_name(cond_name)
            val_unit_val_ast = prod_val_unit_ast(val_unit_val, tabs)
            val_unit_val_field = RealizedField(cond_prod['val_unit_val'], val_unit_val_ast)
            cond_fields = [val_unit_val_field]
            if isinstance(val_1, dict) and op_id_val != 1:
                query_val_ast = prod_query_expr(val_1)
                query_val_field = RealizedField(cond_prod['body'], query_val_ast)
                cond_fields.append(query_val_field)
            return AbstractSyntaxTree(cond_prod, cond_fields)
        elif cond_val[1] == 'and':
            cond_prod = grammar.get_prod_by_ctr_name('And')
            cond_val1 = RealizedField(cond_prod['cond_val1'], prod_cond_ast(cond_val[:1], tabs))
            cond_val2 = RealizedField(cond_prod['cond_val2'], prod_cond_ast(cond_val[2:], tabs))
            return AbstractSyntaxTree(cond_prod, [cond_val1, cond_val2])
        elif cond_val[1] == 'or':
            cond_prod = grammar.get_prod_by_ctr_name('Or')
            cond_val1 = RealizedField(cond_prod['cond_val1'], prod_cond_ast(cond_val[:1], tabs))
            cond_val2 = RealizedField(cond_prod['cond_val2'], prod_cond_ast(cond_val[2:], tabs))
            return AbstractSyntaxTree(cond_prod, [cond_val1, cond_val2])
        else:
            raise ValueError('unknown cond conj op: %s' % cond_val[1])

    def prod_from_cond_ast(cond_val, tabs):
        cond_prod = grammar.get_prod_by_ctr_name('FromCond')
        col_unit1 = prod_col_unit_ast((0, cond_val[0], None), tabs)
        col_unit2 = prod_col_unit_ast((0, cond_val[1], None), tabs)
        col_val1 = RealizedField(cond_prod['col_unit_val1'], col_unit1)
        col_val2 = RealizedField(cond_prod['col_unit_val2'], col_unit2)
        return AbstractSyntaxTree(cond_prod, [col_val1, col_val2])

    def prod_from_query(query_val):
        cond_prod = grammar.get_prod_by_ctr_name('FromQuery')
        query_val_ast = sql_query_to_asdl_ast(query_val, grammar, schema)
        query_val_field = RealizedField(cond_prod['body'], query_val_ast)
        return AbstractSyntaxTree(cond_prod, [query_val_field])

    def prod_val_unit_ast(val_unit_val, tabs):
        unit_op_val, col_unit_val_1, col_unit_val_2 = val_unit_val
        if unit_op_val == 0:  # sanity check
            assert col_unit_val_2 is None
        if unit_op_val != 0:
            unit_op_name = UNIT_OPS[unit_op_val].capitalize()
            val_unit_prod_name = grammar.get_prod_by_ctr_name('ValUnit' + unit_op_name)
        else:
            val_unit_prod_name = grammar.get_prod_by_ctr_name('ValUnit')
        fields_list = []
        col_unit_val_1_ast = prod_col_unit_ast(col_unit_val_1, tabs)
        col_unit_val_1_field = RealizedField(val_unit_prod_name['col_unit_val1'], col_unit_val_1_ast)
        fields_list.append(col_unit_val_1_field)
        if col_unit_val_2:
            col_unit_val_2_ast = prod_col_unit_ast(col_unit_val_2, tabs)
            col_unit_val_2_field = RealizedField(val_unit_prod_name['col_unit_val2'], col_unit_val_2_ast)
            fields_list.append(col_unit_val_2_field)
        return AbstractSyntaxTree(val_unit_prod_name, fields_list)

    def prod_col_unit_ast(col_unit_val, tabs):
        col_unit_prod_name = grammar.get_prod_by_ctr_name('ColUnit')
        agg_op_val, col_id_val, is_distinct = col_unit_val
        assert isinstance(col_id_val, int)  # sanity check
        if col_id_val != 0:
            exp_col_id_val = schema._table['col_ori_to_exp'][col_id_val]
        else:
            if len(tabs) == 0 or len(tabs) > 1:
                exp_col_id_val = 0
            else:
                exp_col_id_val = tabs[0] + 1
        field_list = []
        agg_op_val_ast = prod_agg_op_ast(agg_op_val)
        agg_op_val_field = RealizedField(col_unit_prod_name['agg_op_val'], agg_op_val_ast)
        field_list.append(agg_op_val_field)
        col_id_val_field = RealizedField(col_unit_prod_name['col_id_val'], exp_col_id_val)
        field_list.append(col_id_val_field)
        if len(col_unit_prod_name.fields) == 3:
            is_distinct_prod = grammar.get_prod_by_ctr_name('true' if is_distinct else 'false')
            is_distinct_ast = AbstractSyntaxTree(is_distinct_prod)
            is_distinct_field = RealizedField(col_unit_prod_name['is_distinct'], is_distinct_ast)
            field_list.append(is_distinct_field)
        return AbstractSyntaxTree(col_unit_prod_name, field_list, schema._table['exp_column_names'][exp_col_id_val][0])

    def collect_mentioned_tabs(_sql):
        tabs = set()
        for val_unit in _sql['select'][1]:
            tabs.add(schema._table['column_names_original'][val_unit[1][1][1]][0])
            if val_unit[1][2]:
                tabs.add(schema._table['column_names_original'][val_unit[1][2][1]][0])
        conds = []
        if _sql['where']:
            conds = conds + _sql['where'][::2]
        if _sql['having']:
            conds = conds + _sql['having'][::2]
        for cond in conds:
            val_unit = cond[2]
            col_unit = cond[3]
            tabs.add(schema._table['column_names_original'][val_unit[1][1]][0])
            if val_unit[2]:
                tabs.add(schema._table['column_names_original'][val_unit[2][1]][0])
            if isinstance(col_unit, tuple):
                tabs.add(schema._table['column_names_original'][col_unit[1]][0])
        if _sql['groupBy']:
            for col_unit in _sql['groupBy']:
                tabs.add(schema._table['column_names_original'][col_unit[1]][0])
        if _sql['orderBy']:
            for val_unit in _sql['orderBy'][1]:
                tabs.add(schema._table['column_names_original'][val_unit[1][1]][0])
                if val_unit[2]:
                    tabs.add(schema._table['column_names_original'][val_unit[2][1]][0])
        return tabs

    def prod_query_expr(_sql):
        query_expr_prod = grammar.get_prod_by_ctr_name('Query')
        query_expr_fields = []
        tabs = collect_mentioned_tabs(_sql)
        tabs = list(tabs)
        _table_units = [t[1] for t in _sql['from']['table_units'] if (t[0] == 'table_unit' and t[1] != -1)]
        table_units = [t for t in _table_units if t not in tabs]

        # sel_expr
        sel_expr_prod = grammar.get_prod_by_ctr_name('Select')
        select_body = _sql.pop('select')
        _sql['select'] = None
        select_col_ast_lists = []
        for item in select_body[1]:
            select_col_ast_lists.append(prod_select_col_ast(item, _table_units))
        sel_col_val_field = RealizedField(sel_expr_prod['sel_col_val'], select_col_ast_lists)
        sel_expr_ast = AbstractSyntaxTree(sel_expr_prod, [sel_col_val_field])
        sel_expr_field = RealizedField(query_expr_prod['sel_val'], sel_expr_ast)
        query_expr_fields.append(sel_expr_field)

        # from_expr
        from_expr_prod = grammar.get_prod_by_ctr_name('From')
        from_body = _sql.pop('from')
        _sql['from'] = None
        if from_body['conds']:
            from_conds = []
            for cond in from_body['conds'][::2]:
                tab1 = schema._table['column_names_original'][cond[2][1][1]]
                tab2 = schema._table['column_names_original'][cond[3][1]]
                col1 = cond[2][1][1]
                col2 = cond[3][1]
                if tab1[0] not in (table_units + tabs):
                    for idx, col in enumerate(schema._table['column_names_original']):
                        if col[1] == tab1[1] and (col[0] in table_units):
                            col1 = idx
                            break
                if tab2[0] not in (table_units + tabs):
                    for idx, col in enumerate(schema._table['column_names_original']):
                        if col[1] == tab2[1] and (col[0] in table_units):
                            col2 = idx
                            break
                tmp = (col1, col2)
                from_conds.append(tmp)
            conds = from_conds
            if conds:
                from_conds_list = []
                for cond in conds:
                    from_conds_list.append(prod_from_cond_ast(cond, _table_units))
                cond_val_from_field = RealizedField(from_expr_prod['from_cond_val'], from_conds_list)
                from_expr_ast = AbstractSyntaxTree(from_expr_prod, [cond_val_from_field])
                from_expr_field = RealizedField(query_expr_prod['from_val'], from_expr_ast)
            else:
                from_expr_field = RealizedField(query_expr_prod['from_val'])
        else:
            from_expr_field = RealizedField(query_expr_prod['from_val'])
        query_expr_fields.append(from_expr_field)

        # where_expr
        if _sql['where']:
            where_expr_prod = grammar.get_prod_by_ctr_name('Where')
            where_body = _sql.pop('where')
            _sql['where'] = None
            cond_val_where_ast = prod_cond_ast(where_body, _table_units)
            cond_val_where_field = RealizedField(where_expr_prod['cond_val'], cond_val_where_ast)
            where_expr_ast = AbstractSyntaxTree(where_expr_prod, [cond_val_where_field])
            where_expr_field = RealizedField(query_expr_prod['where_val'], where_expr_ast)
        else:
            where_expr_field = RealizedField(query_expr_prod['where_val'])
        query_expr_fields.append(where_expr_field)

        # having_expr
        if _sql['having']:
            having_expr_prod = grammar.get_prod_by_ctr_name('Having')
            having_body = _sql.pop('having')
            _sql['having'] = None
            cond_val_having_ast = prod_cond_ast(having_body, _table_units)
            cond_val_having_field = RealizedField(having_expr_prod['cond_val'], cond_val_having_ast)
            having_expr_ast = AbstractSyntaxTree(having_expr_prod, [cond_val_having_field])
            having_expr_field = RealizedField(query_expr_prod['having_val'], having_expr_ast)
        else:
            having_expr_field = RealizedField(query_expr_prod['having_val'])
        query_expr_fields.append(having_expr_field)

        # orderBy_expr
        if _sql['orderBy']:
            if _sql['limit']:
                order_by_expr_prod = grammar.get_prod_by_ctr_name('Superlative')
            else:
                order_by_expr_prod = grammar.get_prod_by_ctr_name('Order')
            order_by_body = _sql.pop('orderBy')
            _sql['orderBy'] = None
            order_op_val_order_by_ast = prod_order_op_ast(order_by_body[0])
            order_op_val_order_by_field = RealizedField(order_by_expr_prod['order_op_val'], order_op_val_order_by_ast)
            val_unit_val_list = []
            for val_unit_val in order_by_body[1]:
                val_unit_val_order_by_ast = prod_val_unit_ast(val_unit_val, _table_units)
                val_unit_val_list.append(val_unit_val_order_by_ast)
            val_unit_val_field = RealizedField(order_by_expr_prod['val_unit_val'], val_unit_val_list)
            order_by_expr_ast = AbstractSyntaxTree(
                order_by_expr_prod, [order_op_val_order_by_field, val_unit_val_field])
            order_by_expr_field = RealizedField(query_expr_prod['orderBy_val'], order_by_expr_ast)
        else:
            order_by_expr_field = RealizedField(query_expr_prod['orderBy_val'])
        query_expr_fields.append(order_by_expr_field)

        # groupBy_expr
        if _sql['groupBy']:
            group_by_expr_prod = grammar.get_prod_by_ctr_name('GroupBy')
            group_by_body = _sql.pop('groupBy')
            _sql['groupBy'] = None
            col_unit_val_group_by_ast_list = []
            for col_unit_val in group_by_body:
                col_unit_val_group_by_ast = prod_col_unit_ast(col_unit_val, _table_units)
                col_unit_val_group_by_ast_list.append(col_unit_val_group_by_ast)
            col_unit_val_group_by_field = RealizedField(
                group_by_expr_prod['col_unit_val'], col_unit_val_group_by_ast_list)
            group_by_expr_ast = AbstractSyntaxTree(group_by_expr_prod, [col_unit_val_group_by_field])
            group_by_expr_filed = RealizedField(query_expr_prod['groupBy_val'], group_by_expr_ast)
        else:
            group_by_expr_filed = RealizedField(query_expr_prod['groupBy_val'])
        query_expr_fields.append(group_by_expr_filed)

        return AbstractSyntaxTree(query_expr_prod, query_expr_fields)

    if sql['except']:
        asdl_ast = split_query_by_keywords(sql, 'except')
    elif sql['union']:
        asdl_ast = split_query_by_keywords(sql, 'union')
    elif sql['intersect']:
        asdl_ast = split_query_by_keywords(sql, 'intersect')
    else:  # single query
        single_query_prod = grammar.get_prod_by_ctr_name('SingleQuery')
        single_query_ast = prod_query_expr(sql)
        single_query_field = RealizedField(single_query_prod['body'], single_query_ast)
        asdl_ast = AbstractSyntaxTree(single_query_prod, [single_query_field])
    return asdl_ast
