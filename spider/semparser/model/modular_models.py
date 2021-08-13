# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torch import nn
from semparser.common import registry


@registry.register('model', 'encoder-decoder-parser')
class EncoderDecoderParser(nn.Module):

    def __init__(self, encoder, decoder):
        super(EncoderDecoderParser, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, data_batch):

        encoder_output = self.encoder(data_batch)

        if isinstance(encoder_output, tuple):
            return self.decoder(*encoder_output)
        if isinstance(encoder_output, dict):
            return self.decoder(**encoder_output)

        return self.decoder(encoder_output)

    def decode(self, data_batch):

        encoder_output = self.encoder(data_batch)

        if isinstance(encoder_output, tuple):
            return self.decoder.decode(*encoder_output)
        if isinstance(encoder_output, dict):
            return self.decoder.decode(**encoder_output)

        return self.decoder.decode(encoder_output)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, map_location):
        states = torch.load(path, map_location=map_location)
        self.load_state_dict(states)
