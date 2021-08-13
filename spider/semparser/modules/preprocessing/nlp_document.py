# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import warnings
from typing import List, Union

from semparser.modules.preprocessing import Span, NLPToken
from semparser.modules.preprocessing.entity_recognizers import LinkedMention, Mention
from semparser.modules.preprocessing.temporal.temporal_tagger import TemporalDim, TemporalUnit


class NLPDocument:
    def __init__(
        self,
        nlp_tokens: List[NLPToken],
        temporal_units: List[TemporalUnit],
        linked_mentions: List[LinkedMention],
    ):

        self.nlp_tokens = nlp_tokens
        self.temporal_units = temporal_units
        self.linked_mentions = linked_mentions


def _sort_spans(spans: Union[TemporalUnit, LinkedMention]):
    """
    Sort spans by their start index

    Arguments:
        spans: a list of temporal_units or linked_mentions
    Return: a list of [start_index, end_index, span] after sorting by start index in ascending order
    """

    for i in range(0, len(spans)):
        if isinstance(spans[i], TemporalUnit):
            start_index = spans[i].start_index
            end_index = spans[i].end_index
        elif isinstance(spans[i], LinkedMention):
            start_index = spans[i].mention.start_index
            end_index = spans[i].mention.end_index
        else:
            raise ValueError("Don't support type {} in spans".format(type(spans[i])))
        spans[i] = [start_index, end_index, spans[i]]
    return sorted(spans, key=lambda tuple: tuple[0])


def _create_normalized_content_tokens(
    span: Union[TemporalUnit, LinkedMention],
    original_token: NLPToken,
    start_char_index: int,
) -> (str, List[NLPToken]):
    """
    Get normalized content and create new nlp_tokens from the normalized content
    This is a helper function when doing document normalization

    Arguments:
        span: a temporal_unit or linked_mention that we use to normalize the content
        original_token: the original token that needs to be replaced
        start_char_index: the start index of the normalized content in the newly generated content

    Return:
        a string of normalized content
        a list of tokens to represent the normalized content
    """

    if isinstance(span, TemporalUnit):
        normalized_content = span.body
    else:
        normalized_content = span.entity_candidates[0].entity
    new_tokens = []
    start_index = start_char_index
    bodys = normalized_content.split(" ")
    for body in bodys:
        lemmatized_body = body
        new_tokens.append(
            NLPToken(
                body,
                original_token.pos_tag,
                original_token.tag,
                lemmatized_body,
                start_index,
                start_index + len(body),
            )
        )
        start_index += len(body)
    return normalized_content, new_tokens


def _normalize_by_sorted_spans(
    original_tokens, sorted_spans: List[Union[TemporalUnit, LinkedMention]]
):
    """
    Helper function to normalize document
    It replaces the original token with temporal_unit value and linked mention entity
    Also returns the new linked mentions (with mention as previous entity candidate)
    and new temporal units (with content as previous normalized value)
    """

    token_pointer = 0
    span_pointer = 0
    normalized_tokens = []
    new_linked_mentions = []
    new_temporal_units = []
    while token_pointer < len(original_tokens) and span_pointer < len(sorted_spans):
        span_start_index, span_end_index, span = sorted_spans[span_pointer]
        # get current token
        token = original_tokens[token_pointer]
        # calculate the start_index of normalized_token
        if len(normalized_tokens) == 0:
            new_start_index = 0
        else:
            new_start_index = normalized_tokens[-1].end_index + 1
        # check the position of span and token and create normalized token accordingly
        if span_start_index >= token.start_index and span_start_index < token.end_index:
            normalized_span_content, new_normalized_tokens = _create_normalized_content_tokens(
                span, token, new_start_index
            )
            if isinstance(span, TemporalUnit):
                temporal_unit = TemporalUnit(
                    normalized_span_content,
                    new_normalized_tokens[0].start_index,
                    new_normalized_tokens[-1].end_index,
                    span.dim,
                    [normalized_span_content],
                )
                new_temporal_units.append(temporal_unit)
            else:
                linked_mention = LinkedMention(
                    Mention(
                        normalized_span_content,
                        new_normalized_tokens[0].start_index,
                        new_normalized_tokens[-1].end_index,
                    ),
                    span.entity_candidates[:1],
                )
            normalized_tokens += new_normalized_tokens
            while True and token_pointer < len(original_tokens):
                token = original_tokens[token_pointer]
                if token.start_index >= span_end_index:
                    break
                token_pointer += 1
            span_pointer += 1
        elif span_start_index < token.start_index:
            warnings.warn(
                "there seems to exist overlapped spans between {} and {}".format(
                    sorted_spans[span_pointer], sorted_spans[span_pointer - 1]
                )
            )
            logging.warning(
                "there seems to exist overlapped spans between {} and {}".format(
                    sorted_spans[span_pointer], sorted_spans[span_pointer - 1]
                )
            )
            token_pointer -= 1
        else:
            normalized_tokens.append(
                NLPToken(
                    token.body,
                    token.pos_tag,
                    token.tag,
                    token.lemmatized_body,
                    new_start_index,
                    new_start_index + len(token.body),
                )
            )
            token_pointer += 1

    # fill in remaining tokens
    if len(normalized_tokens) == 0:
        new_start_index = 0
    else:
        new_start_index = normalized_tokens[-1].end_index + 1
    for token in original_tokens[token_pointer:]:
        normalized_tokens.append(
            NLPToken(
                token.body,
                token.pos_tag,
                token.tag,
                token.lemmatized_body,
                new_start_index,
                new_start_index + len(token.body),
            )
        )
        new_start_index += len(token.body) + 1
    return NLPDocument(normalized_tokens, new_temporal_units, new_linked_mentions)


def normalize_doc(
    original_doc, use_temporal_values: bool, use_entity_values: bool
) -> NLPDocument:
    """
    Normalize current document by replacing original tokens temporal_unit value and linked mention entity
    Note that corresponding character-level index will be changed too
    """

    spans = []
    if use_temporal_values:
        spans += original_doc.temporal_units
    if use_entity_values:
        spans += original_doc.linked_mentions
    spans = _sort_spans(spans)
    return _normalize_by_sorted_spans(original_doc.nlp_tokens, spans)
