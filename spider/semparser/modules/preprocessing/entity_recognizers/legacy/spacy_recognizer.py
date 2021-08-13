# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List
from semparser.modules.preprocessing.entity_recognizers import (
    EntityType, Mention, Entity, LinkedMention
)


def spacy_ner_entries_to_linked_mentions(entries: List) -> List[LinkedMention]:
    """
    Convert ners identified by SpaCy to a list of LinkedMention defined above
    [TODO]:currently we don't support entity linking, but only spotting
    so we simply just use the mention as the linked_candidate, later on
    we will expand this simple help function to deal with SpaCy ner and
    introduct our own entity spoter and linker
    """
    linked_mentions = []

    for entry in entries:
        entity_type = entry.label_
        if entity_type != "ORG":
            continue

        start_index = entry.start_char
        end_index = entry.end_char
        mention_body = entry.text

        mention = Mention(mention_body, start_index, end_index)
        entity = Entity("None", mention_body, EntityType.Org)
        linked_mentions.append(LinkedMention(mention, [entity]))

    return linked_mentions
