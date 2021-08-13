# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import ABC, abstractmethod
import re


class TextNormalizer(ABC):

    @abstractmethod
    def normalize(self, text: str) -> str:
        pass

class QuoteNormalizer(TextNormalizer):
    
    def normalize(self, text: str) -> str:
        normalized_text = re.sub('[\"|`]','\'', text)
        return re.sub('\'+','\'', normalized_text)

class IDNormalizer(TextNormalizer):
    
    def normalize(self, text: str) -> str:
        normalized_text = re.sub(r"""(^|\s)[i|I][d|D](,|\s|!|\?|\.|$)""",r'\1ID\2', text)
        return normalized_text
