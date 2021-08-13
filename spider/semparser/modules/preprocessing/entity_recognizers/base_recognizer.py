# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import ABC, abstractmethod
from typing import List
from semparser.modules.preprocessing import Span
import enum


class EntityType(enum.Enum):

    Org = 'organization'
    People = 'people_name'
    Location = 'location'
    Other = 'other'


class Mention(Span):

    def __init__(self, body: str, start_index: int, end_index: int):
        """
        body: mention body
        start_index: character-level index of where the mention starts
        end_index: character-level index of where the mention ends
        """
        super().__init__(body, start_index, end_index)

    def __str__(self):
        return '{(%d, %d) %s}' % (self.start_index, self.end_index, self.body)

    def __repr__(self):
        return self.__str__()


class Entity(object):

    def __init__(self, entity_id: str, entity: str, entity_type: EntityType):
        self.entity_id = entity_id
        self.entity = entity
        self.entity_type = entity_type

    def __str__(self):
        return '{Name: %s, Type: %s}' % (self.entity, self.entity_type)

    def __repr__(self):
        return self.__str__()


class LinkedMention(object):

    def __init__(self, mention: Mention, entity_candidates: List[Entity]):
        self.mention = mention
        self.entity_candidates = entity_candidates

    def __str__(self):
        return 'Mention: %s, Candidates: %s' % (
            self.mention, self.entity_candidates
        )

    def __repr__(self):
        return self.__str__()


class BaseRecognizer(ABC):

    @abstractmethod
    def __call__(self, text: str) -> List[LinkedMention]:
        pass
