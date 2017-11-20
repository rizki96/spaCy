from __future__ import unicode_literals, print_function

from os import path

from ..language import Language
from ..attrs import LANG
from ..deprecated import fix_glove_vectors_loading

# Import language-specific data
from .language_data import *

# create Language subclass
class Bahasa(Language):
    lang = 'id' # ISO code

    class Defaults(Language.Defaults):
        lex_attr_getters = dict(Language.Defaults.lex_attr_getters)
        lex_attr_getters[LANG] = lambda text: 'id'

        # override defaults
        tokenizer_exceptions = TOKENIZER_EXCEPTIONS
        tag_map = TAG_MAP
        stop_words = STOP_WORDS

        #lemma_exc = 
        #lemma_index = 
        #lemma_rules = 

    def __init__(self, **overrides):
        overrides = fix_glove_vectors_loading(overrides)
        Language.__init__(self, **overrides)
