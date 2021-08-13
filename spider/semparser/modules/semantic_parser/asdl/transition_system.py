# Copyright (c) 2020-present, Royal Bank of Canada.
# Copyright (c) 2017-present, Pengcheng Yin
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#################################################################################################
# Code is based on the tranX (https://arxiv.org/abs/1810.02720) implementation
# from https://github.com/pcyin/tranX by Pengcheng Yin
#################################################################################################

class Action(object):
    pass

class ApplyRuleAction(Action):
    def __init__(self, production):
        self.production = production

    def __hash__(self):
        return hash(self.production)

    def __eq__(self, other):
        return isinstance(other, ApplyRuleAction) and self.production == other.production

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return 'ApplyRule[%s]' % self.production.__repr__()

class GenTokenAction(Action):
    def __init__(self, token):
        self.token = token

    def __eq__(self, other):
        return isinstance(other, GenTokenAction) and self.token == other.token

    def is_stop_signal(self):
        return self.token == '</primitive>'

    def __repr__(self):
        return 'GenToken[%s]' % self.token

class ReduceAction(Action):
    def __repr__(self):
        return 'Reduce'

    def __eq__(self, other):
        return isinstance(other, ReduceAction)

class TransitionSystem(object):
    def __init__(self, grammar):
        self.grammar = grammar

    def get_actions(self, asdl_ast):
        actions = []

        parent_action = ApplyRuleAction(asdl_ast.production)
        actions.append(parent_action)

        for field in asdl_ast.fields:
            if self.grammar.is_composite_type(field.type):
                if field.cardinality == 'single':
                    field_actions = self.get_actions(field.value)
                else:
                    field_actions = []

                    if field.value is not None:
                        if field.cardinality == 'multiple':
                            for val in field.value:
                                cur_child_actions = self.get_actions(val)
                                field_actions.extend(cur_child_actions)
                        elif field.cardinality == 'optional':
                            field_actions = self.get_actions(field.value)

                    if field.cardinality == 'multiple' or field.cardinality == 'optional' and not field_actions:
                        field_actions.append(ReduceAction())
            else:
                field_actions = self.get_primitive_field_actions(field)

                if field.cardinality == 'multiple' or field.cardinality == 'optional' and not field_actions:
                    field_actions.append(ReduceAction())

            actions.extend(field_actions)

        return actions

    def tokenize_code(self, code, mode):
        raise NotImplementedError

    def compare_ast(self, hyp_ast, ref_ast):
        raise NotImplementedError

    def ast_to_surface_code(self, asdl_ast):
        raise NotImplementedError

    def surface_code_to_ast(self, code):
        raise NotImplementedError

    def get_primitive_field_actions(self, realized_field):
        raise NotImplementedError

    def get_valid_continuation_types(self, hyp):
        if hyp.tree:
            if self.grammar.is_composite_type(hyp.frontier_field.type):
                if hyp.frontier_field.cardinality == 'single':
                    return ApplyRuleAction,
                else:
                    return ApplyRuleAction, ReduceAction
            else:
                if hyp.frontier_field.cardinality == 'single':
                    return GenTokenAction,
                elif hyp.frontier_field.cardinality == 'optional':
                    if hyp._value_buffer:
                        return GenTokenAction,
                    else:
                        return GenTokenAction, ReduceAction
                else:
                    return GenTokenAction, ReduceAction
        else:
            return ApplyRuleAction,

    def get_valid_continuating_productions(self, hyp):
        if hyp.tree:
            if self.grammar.is_composite_type(hyp.frontier_field.type):
                return self.grammar[hyp.frontier_field.type]
            else:
                raise ValueError
        else:
            return self.grammar[self.grammar.root_type]
