# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from semparser.modules.semantic_parser.asdl.asdl import ASDLCompositeType
from semparser.modules.semantic_parser.asdl.asdl_ast import AbstractSyntaxTree
from semparser.modules.semantic_parser.asdl.transition_system import ApplyRuleAction, ReduceAction, GenTokenAction

class SpiderHypothesis(object):
    def __init__(self, schema):
        self.schema = schema
        self.tree = None
        self.actions = []
        self.score = 0.
        self.frontier_node = None
        self.frontier_field = None

        self.t = 0

    def apply_action(self, action):
        if self.tree is None:
            assert isinstance(action, ApplyRuleAction), 'Invalid action [%s], only ApplyRule action is valid ' \
                'at the beginning of decoding'
            self.tree = AbstractSyntaxTree(action.production)
            self.update_frontier_info()
        elif self.frontier_node:
            if isinstance(self.frontier_field.type, ASDLCompositeType):
                if isinstance(action, ApplyRuleAction):
                    field_value = AbstractSyntaxTree(action.production)
                    field_value.created_time = self.t
                    self.frontier_field.add_value(field_value)
                    self.update_frontier_info()
                elif isinstance(action, ReduceAction):
                    assert self.frontier_field.cardinality in ('optional', 'multiple'), 'Reduce action can only be' \
                        'applied on field with multiple cardinality'
                    self.frontier_field.set_finish()
                    self.update_frontier_info()
                else:
                    raise ValueError('Invalid action [%s] on field [%s]' % (action, self.frontier_field))
            else:
                if isinstance(action, GenTokenAction):
                    self.frontier_field.add_value(action.token)
                    if isinstance(action.token, int):
                        self.frontier_node.additional_info = self.schema._table[
                            'exp_column_names'][action.token][0]

                    if self.frontier_field.cardinality in ('single', 'optional'):
                        self.frontier_field.set_finish()
                        self.update_frontier_info()
                elif isinstance(action, ReduceAction):
                    assert self.frontier_field.cardinality in ('optional', 'multiple'), 'Reduce action can only be' \
                        'applied on field with multiple cardinality'
                    self.frontier_field.set_finish()
                    self.update_frontier_info()
                else:
                    raise ValueError('Can only invoke GenToken or Reduce actions on primitive fields')

        self.t += 1
        self.actions.append(action)

    def update_frontier_info(self):
        def _find_frontier_node_and_field(tree_node):
            if tree_node:
                for field in tree_node.fields:
                    if isinstance(field.type, ASDLCompositeType) and field.value:
                        if field.cardinality in ('single', 'optional'):
                            iter_values = [field.value]
                        else:
                            iter_values = field.value

                        for child_node in iter_values:
                            result = _find_frontier_node_and_field(child_node)
                            if result:
                                return result

                    if not field.finished:
                        return tree_node, field

                return None
            else:
                return None

        frontier_info = _find_frontier_node_and_field(self.tree)
        if frontier_info:
            self.frontier_node, self.frontier_field = frontier_info
        else:
            self.frontier_node, self.frontier_field = None, None

    def clone_and_apply_action(self, action):
        new_hyp = self.copy()
        new_hyp.apply_action(action)

        return new_hyp

    def copy(self):
        new_hyp = SpiderHypothesis()
        if self.tree:
            new_hyp.tree = self.tree.copy()

        new_hyp.actions = list(self.actions)
        new_hyp.score = self.score
        new_hyp.t = self.t

        new_hyp.update_frontier_info()

        return new_hyp

    @property
    def completed(self):
        return self.tree and self.frontier_field is None

class SpiderDecodeHypothesis(SpiderHypothesis):
    def __init__(self, schema):
        super(SpiderDecodeHypothesis, self).__init__(schema)

        self.action_infos = []
        self.code = None

    def clone_and_apply_action_info(self, action_info):
        action = action_info.action

        new_hyp = self.clone_and_apply_action(action)
        new_hyp.action_infos.append(action_info)

        return new_hyp

    def copy(self):
        new_hyp = SpiderDecodeHypothesis(self.schema)
        if self.tree:
            new_hyp.tree = self.tree.copy()

        new_hyp.actions = list(self.actions)
        new_hyp.action_infos = list(self.action_infos)
        new_hyp.score = self.score
        new_hyp.t = self.t
        new_hyp.code = self.code

        new_hyp.update_frontier_info()

        return new_hyp
