# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Preprocessor to get all basic nlp information from given text"""
from abc import ABC, abstractmethod
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex
from semparser.modules.preprocessing.entity_recognizers import spacy_ner_entries_to_linked_mentions
from semparser.modules.preprocessing.nlp_document import NLPDocument, NLPToken
from semparser.modules.preprocessing.temporal.temporal_tagger import RuleBasedTemporalTagger
from semparser.modules.preprocessing.text_normalizers.basic_normalizers import TextNormalizer, QuoteNormalizer, IDNormalizer

from typing import List


SUPPORT_ENTITY_RECOGNIZER_TYPES = ["spacy"]
SUPPORT_TEMPORAL_TAGGER_TYPES = ["rule"]
SUPPORT_TAG_LEMMATIZER_TYPES = ['spacy', 'wordnet']

class NLPPreprocessor(ABC):
    @abstractmethod
    def preprocessing(self, text: str) -> NLPDocument:
        """
        Preprocessing the given text to generate a list of NLPTokens 
        It contains all basic nlp information of a token
        and can be used for downstream tasks without duplicating the work. 
        """


class AlanNLPPreprocessor(NLPPreprocessor):
    """
    NLPPreprocessor that uses SpaCy for tokenization, lemmatization and pos-tagging
    """

    def __init__(self,
        text_normalizers: List[TextNormalizer] = [QuoteNormalizer(), IDNormalizer()],
        entity_recognizer_type: str = None, 
        temporal_tagger_type: str = None,
        tag_and_lemma_type: str = None):

        # prepare for tagging and lemmatization
        self.tag_and_lemma_type = tag_and_lemma_type
        if self.tag_and_lemma_type and (self.tag_and_lemma_type not in SUPPORT_TAG_LEMMATIZER_TYPES):
            raise ValueError(
                "Currently only support pos tagger and corresponding lemmatizer {}, \
                {} not covered".format(
                    SUPPORT_TAG_LEMMATIZER_TYPES, tag_and_lemma_type
                )
            )
        if self.tag_and_lemma_type == 'wordnet':
            self.wordnet_lemmatizer = WordNetLemmatizer()
        # initialize nlp based on entity_recognizer_type
        self.entity_recognizer_type = entity_recognizer_type
        if self.entity_recognizer_type and (self.entity_recognizer_type not in SUPPORT_ENTITY_RECOGNIZER_TYPES):
            raise ValueError(
                "Currently only support ner tagger {}, \
                {} not covered".format(
                    SUPPORT_ENTITY_RECOGNIZER_TYPES, entity_recognizer_type
                )
            )

        self.nlp = self.create_customized_spacy_pipeline(
            self.entity_recognizer_type != 'spacy', 
            self.tag_and_lemma_type != 'spacy')
        # initialize temporal tagger based on temporal_tagger_type
        self.temporal_tagger_type = temporal_tagger_type
        if self.temporal_tagger_type and (self.temporal_tagger_type not in SUPPORT_TEMPORAL_TAGGER_TYPES):
            raise ValueError(
                "Currently only support temporal tagger {}, \
                {} not covered".format(
                    SUPPORT_TEMPORAL_TAGGER_TYPES, temporal_tagger_type
                )
            )
        if self.temporal_tagger_type == "rule":
            self.temporal_tagger = RuleBasedTemporalTagger()
        else:
            self.temporal_tagger = None

        self.text_normalizers = text_normalizers

    def preprocessing(self, text: str) -> NLPDocument:

        # normalize text
        normalized_text = text
        for nor in self.text_normalizers:
            normalized_text = nor.normalize(normalized_text)

        # process text through spacy
        doc = self.nlp(normalized_text)
        nlp_tokens = []
        tags = pos_tags = lemma_bodies = [None] * len(doc)
        if self.tag_and_lemma_type == 'spacy':
            tags = [t.tag_ for t in doc]
            pos_tags = [t.pos_ for t in doc]
            lemma_bodies = [t.lemma_ for t in doc]
        elif self.tag_and_lemma_type == 'wordnet':
            pos_tags = tags = [e[1] for e in nltk.pos_tag([t.text for t in doc])]
            wordnet_pos = [self.get_wornet_pos(e) for e in pos_tags]
            lemma_bodies = [self.wordnet_lemmatizer.lemmatize(x, e) if e else x.lower() for x, e in zip([t.text for t in doc], wordnet_pos)]

        for i in range(0, len(doc)):
            token = doc[i]
            lemma = lemma_bodies[i]
            if token.text.lower() == "ids":
                lemma = "id"
            nlp_token = NLPToken(
                token.text,
                pos_tags[i],
                tags[i],
                lemma,
                token.idx,
                token.idx + len(token.text),
            )
            nlp_tokens.append(nlp_token)

        linked_mentions = []
        if self.entity_recognizer_type == "spacy":
            entries = doc.ents
            linked_mentions = spacy_ner_entries_to_linked_mentions(entries)
        
        temporal_units = []
        if self.temporal_tagger:
            temporal_units = self.temporal_tagger.tag_text(normalized_text)

        return NLPDocument(nlp_tokens, temporal_units, linked_mentions)


    @staticmethod
    def create_customized_spacy_pipeline(disable_ner=False, disable_tagger=False):
        disable = ["parser"]
        if disable_tagger:
            disable.append("tagger")
        if disable_ner:
            disable.append("ner")
        nlp = spacy.load("en_core_web_sm", disable=disable)

        # don't split by '/' or '-'
        infixes = (
            LIST_ELLIPSES
            + LIST_ICONS
            + [
                r"(?<=[0-9])[+\*^](?=[0-9-])",
                r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
                    al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
                ),
                r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
                # EDIT: commented out regex that splits on hyphens between letters:
                #r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
                r"(?<=[{a}0-9])[:<>=](?=[{a}])".format(a=ALPHA),
            ]
        )
        infix_re = compile_infix_regex(infixes)
        nlp.tokenizer.infix_finditer = infix_re.finditer
        return nlp

    @staticmethod
    def get_wornet_pos(pos_tag):
        if pos_tag.startswith('J'):
            return wn.ADJ
        elif pos_tag.startswith('V'):
            return wn.VERB
        elif pos_tag.startswith('N'):
            return wn.NOUN
        elif pos_tag.startswith('R'):
            return wn.ADV
        else:
            return None
