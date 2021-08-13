# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List
from semparser.modules.preprocessing.schema_linker.base_schema_linker import BaseSchemaLinker


class RatSchemaLinker(BaseSchemaLinker):

    def __init__(
        self,
        concept_net_folder = 'data/common/concept_net/',
        wordnet_synonyms_for_table_and_column = False,
        use_temporal_result = True,
        use_entity_result = True,
        use_alter_names = True,
        use_db_value = True,
        token_type_norm = False):

        super().__init__(concept_net_folder, wordnet_synonyms_for_table_and_column, 
            use_temporal_result, use_entity_result,
            use_alter_names, use_db_value)
        self.token_type_norm = token_type_norm
       

    def linking(self, nlp_document) -> List:
        """
        Linking using the same rules in the linking method, return exhausive matching result rather than prioritized matching result
        This result is being used as input of rat-sql model
        0: no match
        1: partial match
        2: full match
        3: value match
        """

        # prepare for linking
        tokens, entity_token_check_dict, temporal_token_check_dict = self._linking_preparation(nlp_document)
        processed_token_bodies = self._get_token_bodies(tokens)
        num_tokens = len(tokens)
        current_idx = 0

        # normalize token
        if self.token_type_norm:
            while current_idx < num_tokens:
                res = self._group_digital([t.body for t in tokens], current_idx)
                if current_idx in temporal_token_check_dict:
                    end_current_idx = temporal_token_check_dict[current_idx]
                    for idx in range(current_idx, end_current_idx):
                        processed_token_bodies[idx] = 'year'
                    current_idx = end_current_idx
                elif res:
                    processed_token_bodies[current_idx]= 'number'
                    current_idx += 1
                else:
                    current_idx += 1

        # update table match types
        tab_types = [[0]* num_tokens for _ in range(len(self.tab_names))] 
        current_idx = 0

        skip_indices =set()

        while current_idx < num_tokens:

            # fully match table names
            return_match = self._full_match(processed_token_bodies, current_idx, self.tab_names)
            tab_types = self._update_types_with_span(return_match, current_idx, tab_types, 2)

            # skip if have multi-gram full match
            if any([idx > current_idx + 1 for idx, _ in return_match.items()]):
                end_index = max(list(return_match.keys()))
                for idx in range(current_idx, end_index):
                    skip_indices.add(idx)
                current_idx = end_index
                continue 
            
            # partial match table names
            return_match = self._rough_partial_match(processed_token_bodies, current_idx, self.tab_name_lists)
            tab_types = self._update_types_with_span(return_match, current_idx, tab_types, 1)

            # synonyms match
            for original_tab_idx, headers in self.tab_synonyms.items():            
                return_match = self._full_match(processed_token_bodies, current_idx, headers)
                return_match = {k: [original_tab_idx] for k, v in return_match.items()}
                tab_types = self._update_types_with_span(return_match, current_idx, tab_types, 2)
            current_idx += 1

        # update column match types
        col_types = [[0]* num_tokens for _ in range(len(self.col_names))] 
        current_idx = 0

        while current_idx < num_tokens:
            # fully match column names
            return_match = self._full_match(processed_token_bodies, current_idx, self.col_names)
            col_types = self._update_types_with_span(return_match, current_idx, col_types, 2)

            # skip if have multi-gram full match
            if any([idx > current_idx + 1 for idx, _ in return_match.items()]):
                current_idx = max(list(return_match.keys()))
                continue   
                               
            # partial match column names
            return_match = self._rough_partial_match(processed_token_bodies, current_idx, self.col_name_lists)
            col_types = self._update_types_with_span(return_match, current_idx, col_types, 1)

            # synonyms match of column names
            for original_col_idx, headers in self.col_synonyms.items():            
                return_match = self._full_match(processed_token_bodies, current_idx, headers)
                return_match = {k: [original_col_idx] for k, v in return_match.items()}
                col_types = self._update_types_with_span(return_match, current_idx, col_types, 2)
            
            
            if current_idx in skip_indices:
                current_idx += 1
                continue

            # identify value based on if it is all digit
            res = self._group_digital(processed_token_bodies, current_idx)
            if res:
                matched_indices = self._get_fixed_span_value_source([processed_token_bodies[current_idx]], False)
                return_match = {current_idx + 1: matched_indices}
                col_types = self._update_types_with_span(return_match, current_idx, col_types, 3)

            # identify value based on if it is quoted
            end_current_idx, span = self._group_symbol(processed_token_bodies, current_idx)
            if span:
                matched_indices = self._get_fixed_span_value_source(span)
                return_match = {end_current_idx: matched_indices}
                col_types = self._update_types_with_span(return_match, current_idx, col_types, 3)

            # identify values based on if it is capitalized
            end_current_idx, span = self._group_values([t.body for t in tokens], current_idx)
            if span:
                matched_indices = self._get_fixed_span_value_source(processed_token_bodies[current_idx:end_current_idx])
                return_match = {end_current_idx: matched_indices}
                col_types = self._update_types_with_span(return_match, current_idx, col_types, 3)

            # identify value based on if it is a fully matched to db categorical content
            if self.use_db_value:
                matched_indices = self._db_categorical_text_value_match(processed_token_bodies, current_idx)
                return_match = {current_idx + 1: matched_indices}
                col_types = self._update_types_with_span(return_match, current_idx, col_types, 3)           
            
            # identify value based on entity information
            if self.use_entity_result and current_idx in entity_token_check_dict:
                end_current_idx = entity_token_check_dict[current_idx]
                matched_indices = self._get_fixed_span_value_source(processed_token_bodies[current_idx:end_current_idx])
                return_match = {end_current_idx: matched_indices}
                col_types = self._update_types_with_span(return_match, current_idx, col_types, 3)
            current_idx += 1

        return (processed_token_bodies,
            [t.body for t in tokens], 
            self.tab_names,
            self.col_names, 
            self.original_table_names,
            self.original_column_names, 
            tab_types, col_types) 

    @staticmethod
    def _update_types_with_span(return_match, current_idx, type_matrix, update_type):
        if not return_match:
            type_matrix 
        for end_current_idx, header_indices in return_match.items():
            for token_index in range(current_idx, end_current_idx):
                for header_index in header_indices:
                    if type_matrix[header_index][token_index] == 0:
                        type_matrix[header_index][token_index] = update_type
        return type_matrix
    
