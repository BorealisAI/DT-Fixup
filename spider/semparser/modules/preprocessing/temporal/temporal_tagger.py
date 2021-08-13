# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import ABC, abstractmethod
from enum import Enum
from typing import List

from semparser.modules.preprocessing import Span
from semparser.modules.preprocessing.tokenizer import SpacyTokenizer, Tokenizer


class TemporalDim(Enum):
    """
    """

    Time = "time"
    Duration = "duration"
    Number = "number"
    AmountOfMoney = "amount-of-money"

TEMPORAL_VALID_VALUES = set([item.value for item in TemporalDim])

class TemporalUnit(Span):
    def __init__(
        self,
        body: str,
        start_index: int,
        end_index: int,
        dim: TemporalDim,
        matched_values: List,
    ):
        super().__init__(body, start_index, end_index)
        self.dim = dim
        self.matched_values = matched_values


class TemporalTagger(ABC):
    @abstractmethod
    def tag_text(self, text: str) -> List[TemporalUnit]:
        pass


class RuleBasedTemporalTagger(TemporalTagger):
    """
    Using naive rules to label time related terms
    (From origianl IRNet(https://github.com/microsoft/IRNet) schema linking code)
    """

    def __init__(self, tokenizer: Tokenizer = SpacyTokenizer()):
        self.tokenizer = tokenizer

    def tag_text(self, text: str) -> List[TemporalUnit]:
        units = []
        tokens = self.tokenizer.tokenize(text)
        for tok in tokens:
            if (
                len(str(tok.body)) == 4
                and str(tok.body).isdigit()
                and int(str(tok.body)[:2]) < 22
                and int(str(tok.body[:2])) > 15
            ):
                temporal_unit = TemporalUnit(
                    tok.body,
                    tok.start_index,
                    tok.end_index,
                    TemporalDim.Time,
                    [{"type": "value", "value": tok.body}],
                )
                units.append(temporal_unit)
        return units
