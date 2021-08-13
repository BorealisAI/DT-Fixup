# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from semparser.common import registry

from semparser.modules.semantic_parser.asdl.transition_system import ApplyRuleAction, ReduceAction
from semparser.modules.semantic_parser.asdl.spider.spider_transition_system import SpiderGenTokenAction, \
    SpiderSelectColumnAction
from semparser.modules.semantic_parser.asdl.spider.spider_hypothesis import SpiderDecodeHypothesis
from semparser.modules.semantic_parser.asdl.action_info import ActionInfo
from semparser.model.pointer_net import PointerNet
from semparser.model.rat_enc_modules import RelativeAttention, get_att_mask_2d


@registry.register('decoder', 'parent-feed-lstm')
class ParentFeedingLSTMDecoder(nn.Module):

    def __init__(self, enc_dim, act_dim, col_dim, type_dim, rnn_dim, attn_dim, col_sel_weight,
                 transition_system, beam_size=5, max_decode_step=100, label_smoothing=0.2,
                 device='cpu', att_dropout=0.5, input_dropout=0.1, prod_temp=1.0, col_temp=1.0,
                 no_input_feed=False,
                 no_parent_production_embed=False,
                 no_parent_type_embed=False,
                 no_parent_state=False,
                 ):
        super(ParentFeedingLSTMDecoder, self).__init__()

        self.production_embed = nn.Embedding(
            len(transition_system.grammar) + 1, act_dim)
        self.type_embed = nn.Embedding(len(transition_system.grammar.types), type_dim)
        self.att_dropout = nn.Dropout(att_dropout)
        self.input_dropout = nn.Dropout(input_dropout)

        self.label_smoothing = label_smoothing
        self.prod_temp = prod_temp
        self.col_temp = col_temp

        input_dim = act_dim
        input_dim += act_dim * (not no_parent_production_embed)
        input_dim += type_dim * (not no_parent_type_embed)
        input_dim += rnn_dim * (not no_parent_state)
        input_dim += attn_dim * (not no_input_feed)

        self.col_sel_weight = col_sel_weight
        self.transition_system = transition_system
        self.grammar = self.transition_system.grammar

        self.decoder_input_dim = input_dim
        self.device = device
        self.act_dim = act_dim
        self.beam_size = beam_size
        self.max_decode_step = max_decode_step

        self.no_input_feed = no_input_feed
        self.no_parent_production_embed = no_parent_production_embed
        self.no_parent_type_embed = no_parent_type_embed
        self.no_parent_state = no_parent_state

        self.decoder_lstm = nn.LSTMCell(input_dim, rnn_dim)

        self.att_vec_linear = nn.Linear(
            enc_dim + rnn_dim, attn_dim, bias=False)
        self.att_over_enc = RelativeAttention(
            rnn_dim, enc_dim, enc_dim)

        self.query_vec_to_action_embed = nn.Sequential(
            nn.Linear(attn_dim, attn_dim * 4),
            nn.Tanh(),
            nn.Linear(attn_dim * 4, act_dim),
        )

        self.linear = nn.Linear(self.production_embed.weight.size(1),
                                self.production_embed.weight.size(0), bias=True)
        self.linear.weight = self.production_embed.weight
        self.production_readout = nn.Sequential(self.query_vec_to_action_embed, self.linear)

        self.column_pointer_net = PointerNet(attn_dim, col_dim, normalized=False)

        self.column_rnn_input = nn.Linear(col_dim, act_dim, bias=False)
        self.column_mem = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, batch, enc, col_enc, enc_lens, col_lens):
        examples = batch.examples
        batch.grammar = self.grammar

        col_mask = batch.column_mask
        pad_mask = get_att_mask_2d([1] * len(enc_lens), enc_lens).to(self.device)

        h_tm1 = None

        col_appear_mask = np.zeros((len(examples), batch.max_col_num), dtype=np.float32)

        zero_action_embed = torch.zeros(self.act_dim, dtype=torch.float, device=self.device)
        history_states = []
        att_vecs = []

        action_probs = [[] for _ in examples]
        att_tm1 = None

        for t in range(batch.max_action_num):
            col_appear_mask_val = torch.from_numpy(col_appear_mask).to(self.device)

            x = self._prep_step_input(batch, examples, col_enc,
                                      zero_action_embed, att_tm1,
                                      history_states, t)

            (h_t, cell_t), att_t = self.step(x, h_tm1, enc, mask=pad_mask)

            apply_rule_log_prob = F.log_softmax(self.production_readout(att_t).div_(self.prod_temp), dim=-1)

            column_attention_log_weights = self.col_pred_prob(
                att_t, col_enc, col_appear_mask_val, column_mask=col_mask)

            #########################################################################################

            for e_id, example in enumerate(examples):
                if t < len(example.tgt_actions):
                    action_info_t = example.tgt_actions[t]
                    action_t = action_info_t.action

                    if isinstance(action_t, ApplyRuleAction):

                        act_log_prob_t_i = apply_rule_log_prob[e_id,
                                                               self.grammar.prod2id[action_t.production]]

                    elif isinstance(action_t, ReduceAction):
                        act_log_prob_t_i = apply_rule_log_prob[e_id, len(self.grammar)]

                    elif isinstance(action_t, SpiderSelectColumnAction):
                        col_appear_mask[e_id, action_t.column_id] = 1.
                        act_log_prob_t_i = column_attention_log_weights[e_id, action_t.column_id]

                        if self.label_smoothing > 0.:
                            act_log_prob_t_i = (1. - self.label_smoothing) * act_log_prob_t_i
                            act_log_prob_t_i = act_log_prob_t_i + \
                                column_attention_log_weights[e_id, :col_lens[e_id]].sum(-1) \
                                * self.label_smoothing / col_lens[e_id]
                    else:
                        raise ValueError('unknown action %s' % action_t)

                    action_probs[e_id].append(act_log_prob_t_i.reshape(1))

            history_states.append((h_t, cell_t))
            att_vecs.append(att_t)

            h_tm1 = (h_t, cell_t)
            att_tm1 = att_t

        action_prob_var = torch.cat([torch.cat(action_probs_i).sum(-1, keepdim=True)
                                     for action_probs_i in action_probs])

        result = action_prob_var.mean()
        if result > 0:
            print('numerical instability:', result.item())

        return -torch.mean(result)

    def decode(self, batch, sent_enc, col_enc, sent_lens, col_lens):

        hyps_batch = list()
        for batch_idx, example in enumerate(batch.examples):

            sent_enc_ = sent_enc[batch_idx:batch_idx + 1, :sent_lens[batch_idx], :]
            col_enc_ = col_enc[batch_idx:batch_idx + 1, :col_lens[batch_idx], :]

            hyps_batch.append(self.decode_single_example(example, sent_enc_, col_enc_))

        return hyps_batch

    def decode_single_example(self, example, sent_enc, col_enc):
        h_tm1 = None
        zero_action_embed = torch.zeros(self.act_dim, device=self.device, dtype=torch.float)

        t = 0
        hypotheses = [SpiderDecodeHypothesis(example.schema)]
        hyp_states = [[]]
        completed_hypotheses = []
        att_tm1 = None
        while len(completed_hypotheses) < self.beam_size and t < self.max_decode_step:
            hyp_num = len(hypotheses)
            exp_enc_padded = sent_enc.expand(hyp_num, sent_enc.size(1), sent_enc.size(2))
            num_cols = col_enc.size()[1]
            col_appear_mask = np.zeros((hyp_num, num_cols), dtype=np.float32)
            for e_id, hyp in enumerate(hypotheses):
                for act in hyp.actions:
                    if isinstance(act, SpiderSelectColumnAction):
                        col_appear_mask[e_id, act.column_id] = 1.

            col_appear_mask_val = torch.from_numpy(col_appear_mask).to(self.device)

            if t == 0:
                x = torch.zeros((1, self.decoder_input_dim), device=self.device, dtype=torch.float)
            else:
                a_tm1_embeds = []
                for e_id, hyp in enumerate(hypotheses):
                    action_tm1 = hyp.actions[-1]
                    if action_tm1:
                        if isinstance(action_tm1, ApplyRuleAction):
                            a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]
                        elif isinstance(action_tm1, ReduceAction):
                            a_tm1_embed = self.production_embed.weight[len(self.grammar)]
                        elif isinstance(action_tm1, SpiderSelectColumnAction):
                            a_col_embed = col_enc[0, action_tm1.column_id]
                            a_tm1_embed = self.column_rnn_input(a_col_embed)
                        elif isinstance(action_tm1, SpiderGenTokenAction):
                            a_tm1_embed = zero_action_embed
                        else:
                            raise ValueError('unknown action %s' % action_tm1)
                    else:
                        a_tm1_embed = zero_action_embed

                    a_tm1_embeds.append(a_tm1_embed)

                a_tm1_embeds = torch.stack(a_tm1_embeds)

                inputs = [a_tm1_embeds]
                if self.no_input_feed is False:
                    inputs.append(att_tm1)
                if self.no_parent_production_embed is False:
                    frontier_prods = [hyp.frontier_node.production for hyp in hypotheses]
                    frontier_prod_embeds = self.production_embed(torch.tensor(
                        [self.grammar.prod2id[prod] for prod in frontier_prods], device=self.device, dtype=torch.long))
                    inputs.append(frontier_prod_embeds)
                if self.no_parent_type_embed is False:
                    frontier_types = [hyp.frontier_field.type for hyp in hypotheses]
                    frontier_type_embeds = self.type_embed(torch.tensor(
                        [self.grammar.type2id[type] for type in frontier_types], device=self.device, dtype=torch.long))
                    inputs.append(frontier_type_embeds)

                if self.no_parent_state is False:
                    p_ts = [hyp.frontier_node.created_time for hyp in hypotheses]
                    parent_states = torch.stack([hyp_states[hyp_id][p_t][0]
                                                 for hyp_id, p_t in enumerate(p_ts)])
                    inputs.append(parent_states)

                x = torch.cat(inputs, dim=-1)

            (h_t, cell_t), att_t = self.step(x, h_tm1, exp_enc_padded)

            apply_rule_log_prob = F.log_softmax(self.production_readout(att_t).div_(self.prod_temp), dim=-1)

            gate = torch.sigmoid(self.column_mem(att_t))
            w_c = self.column_pointer_net(col_enc, None, att_t.unsqueeze(0)).squeeze(0)

            c_schema_mask = col_appear_mask_val
            c_memory_mask = (1 - col_appear_mask_val)

            weights = (w_c * c_schema_mask * gate + w_c * c_memory_mask * (1 - gate))

            column_selection_log_prob = F.log_softmax(weights.div_(self.col_temp), dim=-1)

            new_hyp_meta = []

            for hyp_id, hyp in enumerate(hypotheses):
                action_types = self.transition_system.get_valid_continuation_types(hyp)

                for action_type in action_types:
                    if action_type == ApplyRuleAction:
                        productions = self.transition_system.get_valid_continuating_productions(hyp)
                        for production in productions:
                            prod_id = self.grammar.prod2id[production]
                            prod_score = apply_rule_log_prob[hyp_id, prod_id]
                            new_hyp_score = hyp.score + prod_score

                            meta_entry = {'action_type': 'apply_rule', 'prod_id': prod_id,
                                          'score': prod_score, 'new_hyp_score': new_hyp_score,
                                          'prev_hyp_id': hyp_id}
                            new_hyp_meta.append(meta_entry)
                    elif action_type == ReduceAction:
                        action_score = apply_rule_log_prob[hyp_id, len(self.grammar)]
                        new_hyp_score = hyp.score + action_score

                        meta_entry = {'action_type': 'apply_rule', 'prod_id': len(self.grammar),
                                      'score': action_score, 'new_hyp_score': new_hyp_score,
                                      'prev_hyp_id': hyp_id}
                        new_hyp_meta.append(meta_entry)
                    elif action_type == SpiderSelectColumnAction:
                        for col_id, column in enumerate(example.origin_col_names):
                            col_sel_score = column_selection_log_prob[hyp_id, col_id]
                            new_hyp_score = hyp.score + self.col_sel_weight * col_sel_score

                            meta_entry = {'action_type': 'sel_col', 'col_id': col_id,
                                          'score': col_sel_score, 'new_hyp_score': new_hyp_score,
                                          'prev_hyp_id': hyp_id}
                            new_hyp_meta.append(meta_entry)

            if not new_hyp_meta:
                break

            new_hyp_scores = torch.cat([x['new_hyp_score'].reshape(1).cpu() for x in new_hyp_meta])
            top_new_hyp_scores, meta_ids = torch.topk(
                new_hyp_scores, k=min(new_hyp_scores.size(0), self.beam_size - len(completed_hypotheses)))

            live_hyp_ids = []
            new_hypotheses = []
            for new_hyp_score, meta_id in zip(top_new_hyp_scores.data.cpu(), meta_ids.data.cpu()):
                action_info = ActionInfo()
                hyp_meta_entry = new_hyp_meta[meta_id]
                prev_hyp_id = hyp_meta_entry['prev_hyp_id']
                prev_hyp = hypotheses[prev_hyp_id]

                action_type_str = hyp_meta_entry['action_type']
                if action_type_str == 'apply_rule':
                    prod_id = hyp_meta_entry['prod_id']
                    if prod_id < len(self.grammar):
                        production = self.grammar.id2prod[prod_id]
                        action = ApplyRuleAction(production)
                    else:
                        action = ReduceAction()
                elif action_type_str == 'sel_col':
                    action = SpiderSelectColumnAction(hyp_meta_entry['col_id'])

                action_info.action = action
                action_info.t = t

                if t > 0:
                    action_info.parent_t = prev_hyp.frontier_node.created_time
                    action_info.frontier_prod = prev_hyp.frontier_node.production
                    action_info.frontier_field = prev_hyp.frontier_field.field

                new_hyp = prev_hyp.clone_and_apply_action_info(action_info)
                new_hyp.score = new_hyp_score

                if new_hyp.completed:
                    completed_hypotheses.append(new_hyp)
                else:
                    new_hypotheses.append(new_hyp)
                    live_hyp_ids.append(prev_hyp_id)

            if live_hyp_ids:
                hyp_states = [hyp_states[i] + [(h_t[i], cell_t[i])] for i in live_hyp_ids]
                h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
                att_tm1 = att_t[live_hyp_ids]
                hypotheses = new_hypotheses
                t += 1
            else:
                break

        completed_hypotheses.sort(key=lambda hyp: -hyp.score)

        return completed_hypotheses

    def _prep_step_input(self, batch, examples, col_encodings,
                         zero_action_embed, att_tm1, history_states, t):
        if t == 0:
            x = torch.zeros((len(examples), self.decoder_input_dim), dtype=torch.float, device=self.device)
        else:
            a_tm1_embeds = []
            for e_id, example in enumerate(examples):
                if t < len(example.tgt_actions):
                    action_info_tm1 = example.tgt_actions[t - 1]
                    action_tm1 = action_info_tm1.action
                    if isinstance(action_tm1, ApplyRuleAction):
                        a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[action_tm1.production]]
                    elif isinstance(action_tm1, ReduceAction):
                        a_tm1_embed = self.production_embed.weight[len(self.grammar)]
                    elif isinstance(action_tm1, SpiderSelectColumnAction):
                        a_tm1_embed = self.column_rnn_input(
                            col_encodings[e_id, action_tm1.column_id])
                    else:
                        raise ValueError('unknown action %s' % action_tm1)
                else:
                    a_tm1_embed = zero_action_embed

                a_tm1_embeds.append(a_tm1_embed)

            a_tm1_embeds = torch.stack(a_tm1_embeds)

            inputs = [a_tm1_embeds]
            if self.no_input_feed is False:
                inputs.append(att_tm1)
            if self.no_parent_production_embed is False:
                parent_production_embed = self.production_embed(batch.get_frontier_prod_idx(t))
                inputs.append(parent_production_embed)
            if self.no_parent_type_embed is False:
                parent_type_embed = self.type_embed(batch.get_frontier_type_idx(t))
                inputs.append(parent_type_embed)

            actions_t = [e.tgt_actions[t] if t < len(
                e.tgt_actions) else None for e in batch.examples]
            if self.no_parent_state is False:
                parent_states = torch.stack([history_states[p_t][0][batch_id]
                                             for batch_id, p_t in
                                             enumerate(a_t.parent_t if a_t else 0 for a_t in actions_t)])
                inputs.append(parent_states)

            x = torch.cat(inputs, dim=-1)
        return x

    def col_pred_prob(self, att_t, col_encodings, col_appear_mask_val, column_mask=None):
        gate = torch.sigmoid(self.column_mem(att_t))

        w_c = self.column_pointer_net(col_encodings, None, att_t.unsqueeze(0)).squeeze(0)
        c_schema_mask = col_appear_mask_val
        c_memory_mask = 1 - col_appear_mask_val

        weights = (w_c * c_schema_mask * gate + w_c * c_memory_mask * (1 - gate))

        if column_mask is not None:
            weights = weights.masked_fill(column_mask, -float('inf'))
        column_attention_log_weights = F.log_softmax(weights.div_(self.col_temp), dim=-1)

        return column_attention_log_weights

    def step(self, x, h_tm1, enc, enc_att_linear=None, mask=None):

        x = self.input_dropout(x)
        h_t, cell_t = self.decoder_lstm(x, h_tm1)
        ctx_t = self.att_over_enc(h_t.unsqueeze(1), enc, mask).squeeze(1)
        att_t = torch.tanh(self.att_vec_linear(torch.cat([h_t, ctx_t], 1)))
        att_t = self.att_dropout(att_t)

        return (h_t, cell_t), att_t
