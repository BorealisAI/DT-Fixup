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

from semparser.modules.semantic_parser.asdl.transition_system import GenTokenAction

class ActionInfo(object):
    def __init__(self, action=None):
        self.t = 0
        self.parent_t = -1
        self.action = action
        self.frontier_prod = None
        self.frontier_field = None

        self.copy_from_src = False
        self.src_token_position = -1

    def __repr__(self, verbose=False):
        repr_str = '%s (t=%d, p_t=%d, frontier_field=%s)' % (
            repr(self.action), self.t, self.parent_t,
            self.frontier_field.__repr__(True) if self.frontier_field else 'None')
        if verbose:
            verbose_repr = 'action_prob=%.4f, ' % self.action_prob
            if isinstance(self.action, GenTokenAction):
                verbose_repr += 'in_vocab=%s, ' \
                    'gen_copy_switch=%s, ' \
                    'p(gen)=%s, p(copy)=%s, ' \
                    'has_copy=%s, copy_pos=%s' % (self.in_vocab,
                                                  self.gen_copy_switch,
                                                  self.gen_token_prob, self.copy_token_prob,
                                                  self.copy_from_src, self.src_token_position)
            repr_str += '\n' + verbose_repr

        return repr_str
