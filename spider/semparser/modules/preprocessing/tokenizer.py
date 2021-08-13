# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import re
from abc import ABC, abstractmethod
from typing import List
from semparser.modules.preprocessing import NLPToken
import spacy

RE_TOKEN_BOUNDARY = re.compile(r"\w+|\W")


class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> List[NLPToken]:
        pass


class SpacyTokenizer(Tokenizer):
    """NLP text tokenizer using SpaCy"""

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "ner"])

    def tokenize(self, text):
        doc = self.nlp(text)
        tokens = [t for t in doc if t.text.strip() != ""]
        return [NLPToken(t.text, None, None, None, t.idx, t.idx + len(t)) for t in tokens]


class RegexTokenizer(Tokenizer):
    """NLP text tokenizer using naive rules"""

    def __int__(self):
        pass

    def tokenize(self, text):
        tokens = [t for t in RE_TOKEN_BOUNDARY.finditer(text)]
        tokens = [
            NLPToken(t.group(0), None,None, None, t.span()[0], t.span()[1])
            for t in tokens
            if t.group(0).strip() != ""
        ]
        return tokens
