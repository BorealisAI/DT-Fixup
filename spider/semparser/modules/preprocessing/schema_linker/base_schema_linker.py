# Copyright (c) 2020-present, Royal Bank of Canada.
# Copyright (c) Microsoft Corporation.
# All rights reserved.
#
# This source code is licensed under the license found in the
#################################################################################################
# Original code is based on the IRNet (https://arxiv.org/pdf/1905.08205.pdf) implementation
# from https://github.com/microsoft/IRNet/blob/master/preprocess/utils.py
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#################################################################################################

from abc import ABC, abstractmethod
import re
from semparser.modules.preprocessing.nlp_preprocessor import AlanNLPPreprocessor
import pickle
import os
from nltk.corpus import wordnet
from typing import List
from semparser.modules.preprocessing.entity_recognizers import EntityType
from semparser.modules.preprocessing.temporal.temporal_tagger import TemporalDim
from semparser.modules.preprocessing.text_normalizers.basic_normalizers import IDNormalizer
from collections import defaultdict
from semparser.modules.alanschema.schema_with_meta import ColumnWithMeta, TableWithMeta
from semparser.modules.preprocessing.nlp_document import normalize_doc

VALUE_FILTER = ['what', 'how', 'list', 'give', 'show', 'find', 'id', 'order', 'when']
VALUE_ENTITY_TYPES = [EntityType.Org]
VALUE_TEMPORAL_TYPES = [TemporalDim.Time, TemporalDim.Number,
                        TemporalDim.Duration, TemporalDim.AmountOfMoney]
AGG = ['average', 'sum', 'max', 'min', 'minimum', 'maximum', 'between']
TAB_COL_FILTER = ['a', 'an', 'of', 'be', '\'s', '', 'is', 'me', 'it', 'as', 'in', 'on', "do", "to", 'ok',
                  'or', 'at', 'for', 'the', 'and', 'how', 'are', 'out', "all", 'with', 'have', 'that', 'this', 'what']


class BaseSchemaLinker(ABC):

    def __init__(
            self,
            concept_net_folder='data/common/concept_net/',
            wordnet_synonyms_for_table_and_column=False,
            use_temporal_result=True,
            use_entity_result=True,
            use_alter_names=True,
            use_db_value=True):

        # load conceptNet
        with open(os.path.join(concept_net_folder, "english_RelatedTo.pkl"), 'rb') as f:
            self.english_related_to = pickle.load(f)
        with open(os.path.join(concept_net_folder, "english_IsA.pkl"), 'rb') as f:
            self.english_is_a = pickle.load(f)

        # set other configurations
        self.wordnet_synonyms_for_table_and_column = wordnet_synonyms_for_table_and_column
        self.use_temporal_result = use_temporal_result
        self.use_entity_result = use_entity_result
        self.use_alter_names = use_alter_names
        self.use_db_value = use_db_value

    def set_schema(
            self,
            tables_with_meta: List[TableWithMeta],
            columns_with_meta: List[ColumnWithMeta],
            nlp_preprocessor=AlanNLPPreprocessor([IDNormalizer()])):
        """
        Record and preprocess the schema
        """

        self.nlp_preprocessor = nlp_preprocessor
        self.columns_with_meta = columns_with_meta
        self.tables_with_meta = tables_with_meta

        # record original schema
        self.original_table_names = [tab.name for tab in tables_with_meta]
        self.original_column_names = [col.name for col in columns_with_meta]

        # preprocess table names
        self.tab_names = []
        for name in self.original_table_names:
            nlp_tokens = normalize_doc(nlp_preprocessor.preprocessing(name), True, True).nlp_tokens
            self.tab_names.append(" ".join(self._get_token_bodies(nlp_tokens)))

        # preprocess col names
        self.col_names = []
        for name in self.original_column_names:
            nlp_tokens = normalize_doc(nlp_preprocessor.preprocessing(name), True, True).nlp_tokens
            self.col_names.append(" ".join(self._get_token_bodies(nlp_tokens)))
        self.col_name_lists = [re.split(" ", x) for x in self.col_names]
        self.tab_name_lists = [re.split(" ", x) for x in self.tab_names]

        # get synonyms based on input settings
        self.tab_synonyms = self._get_synonyms(self.use_alter_names,
                                               self.wordnet_synonyms_for_table_and_column,
                                               self.tab_names, self.tables_with_meta,
                                               nlp_preprocessor)
        self.col_synonyms = self._get_synonyms(self.use_alter_names,
                                               self.wordnet_synonyms_for_table_and_column,
                                               self.col_names, self.columns_with_meta,
                                               nlp_preprocessor)

    def get_schema(self):
        """
        Print out and return original table names and column names used by the schema linker
        """

        print("Table names are: {}".format(self.original_table_names))
        print("Column names are: {}".format(self.original_column_names))
        return self.original_table_names, self.original_column_names

    def _get_token_bodies(self, nlp_tokens):
        """
        Get token body based on the linker setting
        """

        tokens = []
        for t in nlp_tokens:
            if t.tag.startswith("V") or t.tag.startswith("N"):
                tokens.append(t.lemmatized_body)
            else:
                tokens.append(t.body)
        tokens = [t.lower() for t in tokens]
        return tokens

    @staticmethod
    def _get_token_indices_in_spans(spans, nlp_tokens):
        """
        Return indices of tokens that are within the given spans based on the character-level index
        """

        span_pointer = 0
        token_pointer = 0
        indices = []
        span_token_indices = []
        while token_pointer < len(nlp_tokens) and span_pointer < len(spans):
            span = spans[span_pointer]
            token = nlp_tokens[token_pointer]
            if token.end_index <= span.start_index:
                token_pointer += 1
            elif token.start_index >= span.end_index:
                span_pointer += 1
                if span_token_indices != []:
                    indices.append(span_token_indices)
                span_token_indices = []
            else:
                span_token_indices.append(token_pointer)
                token_pointer += 1
        if span_token_indices != []:
            indices.append(span_token_indices)
        return indices

    @staticmethod
    def _get_value_token_indices_by_mentions(linked_mentions, nlp_tokens):
        """
        Get indices of tokens that are considered as value based on mention information
        """

        spans = [
            item.mention for item in linked_mentions if item.entity_candidates[0].entity_type in VALUE_ENTITY_TYPES]
        return BaseSchemaLinker._get_token_indices_in_spans(spans, nlp_tokens)

    @staticmethod
    def _get_value_token_indices_by_temporal_units(temporal_units, nlp_tokens):
        """
        Get indices of tokens that are considered as value based on temporal information
        - temporal units that are identified as time related (date, time interval etc)
        """

        spans = [item for item in temporal_units if item.dim in VALUE_TEMPORAL_TYPES]
        return BaseSchemaLinker._get_token_indices_in_spans(spans, nlp_tokens)

    def _linking_preparation(self, nlp_document):
        tokens = nlp_document.nlp_tokens

        # get indices of tokens that can be tagged as value based on entity information
        entity_token_indices = self._get_value_token_indices_by_mentions(
            nlp_document.linked_mentions, tokens
        )
        # turn into start_token_index:end_token_index for easy check up
        entity_token_check_dict = {list_[0]: list_[-1] + 1 for list_ in entity_token_indices}
        # get indices of tokens that can be tagged as value based on temporal information
        temporal_token_indices = self._get_value_token_indices_by_temporal_units(
            nlp_document.temporal_units, tokens
        )
        # turn into start_token_index:end_token_index for easy check up
        temporal_token_check_dict = {list_[0]: list_[-1] + 1 for list_ in temporal_token_indices}

        return tokens, entity_token_check_dict, temporal_token_check_dict

    @staticmethod
    def _full_match(toks, current_idx, headers, min_len=1, max_len=None):
        matched_result = defaultdict(list)
        end_index = len(toks) + 1
        if max_len:
            end_index = current_idx + max_len + 1
        for end_current_idx in reversed(range(current_idx + min_len, end_index)):
            sub_toks = toks[current_idx:end_current_idx]
            if len(sub_toks) >= min_len:
                sub_str = " ".join(sub_toks)
                for idx, header in enumerate(headers):
                    if sub_str == header:
                        matched_result[end_current_idx].append(idx)
        return {k: v for k, v in matched_result.items() if v}

    @staticmethod
    def _word_match(toks, current_idx, headers):
        mathced_header_indices = []
        if len(toks[current_idx]) < 2:
            return False
        if toks[current_idx] in TAB_COL_FILTER:
            return False
        for idx, header_toks in enumerate(headers):
            if toks[current_idx] in header_toks:
                return True
        return False

    @staticmethod
    def _rough_partial_match(toks, current_idx, header_toks):
        def is_in(l1, l2):
            l1_str = " ".join(l1)
            l2_str = " ".join(l2)
            if len(l1_str) < 2:
                return False
            return l1_str in l2_str and l1_str not in TAB_COL_FILTER
        matched_result = defaultdict(list)
        end_index = len(toks)
        for idx, header in enumerate(header_toks):
            end_index = min(current_idx + len(header) + 1, len(toks) + 1)
            for end_current_idx in reversed(range(current_idx + 1, end_index)):
                sub_toks = toks[current_idx:end_current_idx]
                if is_in(sub_toks, header):
                    matched_result[end_current_idx].append(idx)
        return {k: v for k, v in matched_result.items() if v}

    @staticmethod
    def _group_values(toks, current_idx):
        def check_capital(tok_list):
            for tok in tok_list:
                if not tok[0].isupper():
                    return False
            return True

        for end_current_idx in reversed(range(current_idx + 1, len(toks) + 1)):
            sub_toks = toks[current_idx:end_current_idx]
            if len(sub_toks) > 1 and check_capital(sub_toks):
                return end_current_idx, sub_toks
            if len(sub_toks) == 1:
                if sub_toks[0][0].isupper() and sub_toks[0].lower() not in VALUE_FILTER \
                        and sub_toks[0].lower().isalnum():
                    return end_current_idx, sub_toks
        return current_idx, None

    @staticmethod
    def _group_symbol(toks, current_idx):
        if toks[current_idx - 1] == "'":
            for i in range(min(3, len(toks) - current_idx)):
                if toks[i + current_idx] == "'":
                    return i + current_idx, toks[current_idx:current_idx + i]
        return current_idx, None

    @staticmethod
    def _group_digital(toks, current_idx):
        tmp = toks[current_idx].replace(':', '')
        tmp = tmp.replace('.', '')
        if tmp and tmp[0] == '-':
            tmp = tmp[1:]
        if tmp.isdigit():
            return True
        else:
            return False

    @staticmethod
    def _get_wordnet_synonyms(headers):
        synonyms = {}
        for header_idx, header in enumerate(headers):
            syns = set()
            for item in wordnet.synsets(header):
                lemma_list = item.lemmas()
                synonyms_names = [item.name().lower() for item in lemma_list]
                syns.update(synonyms_names)
            if header in syns:
                syns.remove(header)
            synonyms[header_idx] = syns
        return synonyms

    def _get_synonyms(self,
                      use_alter_names: bool,
                      use_wordnet: bool,
                      cleaned_headers: list,
                      corresponding_metas: list,
                      nlp_preprocessor):
        synonyms = defaultdict(list)
        if use_wordnet:
            for header_idx, syns in BaseSchemaLinker._get_wordnet_synonyms(cleaned_headers).items():
                synonyms[header_idx] += syns
        if use_alter_names:
            for header_idx, meta in enumerate(corresponding_metas):
                cleaned_alter_names = []
                for name in meta.alter_names:
                    name = " ".join(name.split("_"))
                    nlp_tokens = normalize_doc(
                        nlp_preprocessor.preprocessing(name), True, True).nlp_tokens
                    cleaned_alter_names.append(" ".join(self._get_token_bodies(nlp_tokens)))
                synonyms[header_idx] += cleaned_alter_names
        return {k: v for k, v in synonyms.items() if v}

    def _db_categorical_text_value_match(self, toks, current_idx):
        matched_indices = []
        for col_idx, meta in enumerate(self.columns_with_meta):
            if meta.value_type == "text" and meta.categorical_options:
                if self._word_match(toks, current_idx, [t.split(" ") for t in meta.categorical_options]):
                    matched_indices.append(col_idx)
        return matched_indices

    def _db_number_value_range_match(self, tok):
        # only consider unigram, so there will be only one element
        matched_indices = []
        try:
            number_value = float(tok)
            for col_idx, meta in enumerate(self.columns_with_meta):
                if meta.value_type == "number" and meta.value_range:
                    min_value = meta.value_range["mean"] - \
                        VALUE_VAR_SCALER * meta.value_range["var"]
                    max_value = meta.value_range["mean"] + \
                        VALUE_VAR_SCALER * meta.value_range["var"]
                    if number_value >= min_value and number_value <= max_value:
                        matched_indices.append(col_idx)
            return matched_indices
        except:
            return matched_indices

    @staticmethod
    def _get_concept_result(toks, headers, graph):
        matched_indices = []
        for i in range(len(toks)):
            for j in reversed(range(1, len(toks) + 1 - i)):
                tmp = '_'.join(toks[i:j])
                if tmp in graph:
                    mi = graph[tmp]
                    for idx, header in enumerate(headers):
                        if header in mi and idx not in matched_indices:
                            matched_indices.append(idx)
        return matched_indices

    def _get_fixed_span_value_source(self, symbol: List[str], use_concept_net=True):
        """
        Once identified the fixed span which is a sub list of tokens in the original question,
        this function can be called to help identify columns where the value comes from
        """

        target_column_indices = []
        if self.use_db_value:
            for end_idx in range(0, len(symbol)):
                categorical_matched_indices = self._db_categorical_text_value_match(symbol, end_idx)
                target_column_indices += categorical_matched_indices
        if use_concept_net:
            concept_matched_indices = self._get_concept_result(
                symbol, self.col_names, self.english_is_a)
            if not concept_matched_indices:
                concept_matched_indices = self._get_concept_result(
                    symbol, self.col_names, self.english_related_to)
            target_column_indices += concept_matched_indices
        target_column_indices = list(set(target_column_indices))
        return list(set(target_column_indices))

    @abstractmethod
    def linking(self, nlp_document) -> List:
        pass
