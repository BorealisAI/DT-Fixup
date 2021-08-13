# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

class Span:
    def __init__(self, 
        body: str,
        start_index: int,
        end_index: int):

        self.body = body
        self.start_index = start_index
        self.end_index = end_index


class NLPToken(Span):
    def __init__(
        self,
        body: str,
        pos_tag: str,
        tag: str,
        lemmatized_body: str,
        start_index: int,
        end_index: int,
    ):
        """
        start_index: character-level index where the token starts
        end_index: character-level index where the token ends
        """
        super().__init__(body, start_index, end_index)
        self.pos_tag = pos_tag
        self.tag = tag
        self.lemmatized_body = lemmatized_body
