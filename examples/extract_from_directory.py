"""
Takes al files in DOCS_PATH and extracts entities from them using
the thesaurus specified by PP_PID in the instance given by PP_SERVER.



"""

import logging

from decouple import AutoConfig

from pp_vectorizer import pp_vectorizer as ppv
from pp_vectorizer.doc_organizer import MultilabelDocOrganizer, TextFileIterator

CONFIG = AutoConfig()
# --- Parameters
base_folder = CONFIG("DOCS_PATH")
vectorizer_parameters = {
    'ngram_range': (1, 3),
    'max_df': 0.51,
    'max_features': 1000,
    'use_terms': False,
    'related_prefix': None}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
print('Preparing for extraction')
text_iter = TextFileIterator(base_folder)
print('Text Iterator Prepared')
vectorizer = ppv.PPVectorizer(**vectorizer_parameters)
print('Vectorizer Prepared with params: {}'.format(vectorizer_parameters))
X = vectorizer.fit_transform(text_iter)

