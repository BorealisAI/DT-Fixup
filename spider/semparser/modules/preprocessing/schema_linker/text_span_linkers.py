# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import re
from nltk.stem import WordNetLemmatizer


def link_text_spans_by_token_overlap(src_txt, qry_txt, rp_score=0.5):
    """

    @param src_txt: str
    @param qry_txt: str
    @param rp_score: float
        Recall-Precision score. Range: [0-1]. Higher score lead to higher precision and lower recall.
    @return: bool
    """
    src_list = src_txt.split(' ')
    qry_list = qry_txt.split(' ')

    src_list = [_drop_special_tail_char(src_txt_span) for src_txt_span in src_list]
    if len(src_list) == 0:
        return False

    qry_list = [_drop_special_tail_char(qry_txt_span) for qry_txt_span in qry_list]

    # normalize
    lemmatizer = WordNetLemmatizer()
    src_list = [lemmatizer.lemmatize(token) for token in src_list]
    qry_list = [lemmatizer.lemmatize(token) for token in qry_list]

    src_set = set(src_list)
    src_snt = ' '.join(src_list)
    qry_snt = ' '.join(qry_list)

    overlap = fuzz.token_set_ratio(qry_snt, src_snt) / 100.0
    if overlap >= rp_score:
        return True

    return False


def _drop_special_tail_char(text):
    tail_chars_to_drop = ".,?!"
    for tail_char in tail_chars_to_drop:
        if text.endswith(tail_char):
            text = text[:-1]
            break

    return text.strip()

