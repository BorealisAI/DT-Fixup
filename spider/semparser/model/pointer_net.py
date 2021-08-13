# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointerNet(nn.Module):
    def __init__(self, query_vec_size, src_encoding_size, normalized=True):
        super(PointerNet, self).__init__()
        self.src_encoding_linear = nn.Linear(src_encoding_size, query_vec_size, bias=False)
        self.normalized = normalized

    def forward(self, src_encodings, src_token_mask, query_vec):
        src_encodings = self.src_encoding_linear(src_encodings)
        src_encodings = src_encodings.unsqueeze(1)

        q = query_vec.permute(1, 0, 2).unsqueeze(3)

        weights = torch.matmul(src_encodings, q).squeeze(3)

        weights = weights.permute(1, 0, 2)

        if src_token_mask is not None:
            src_token_mask = src_token_mask.unsqueeze(0).expand_as(weights)
            weights = weights.masked_fill(src_token_mask, -float('inf'))

        if self.normalized is True:
            ptr_weights = F.softmax(weights, dim=-1)
        elif self.normalized == 'log_prob':
            ptr_weights = F.log_softmax(weights, dim=-1)
        else:
            ptr_weights = weights

        return ptr_weights
