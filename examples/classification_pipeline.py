from pp_vectorizer import pp_vectorizer as ppv
from pp_vectorizer.doc_organizer import DocumentGroupVectorizer

import os
import numpy as np
import math

# --- Parameters
ngram_range = (1, 3)
min_term_usage = 0.01
max_term_usage = 0.4

base_folder = os.getenv("DOCS_PATH")

pp_vect = ppv.PPVectorizer(ngram_range=ngram_range)
dfv = DocumentGroupVectorizer(pp_vect, base_folder)
dfv.fit_transform()
pp_vect.cache_extractor.save_cache()

X = dfv.get_whole_matrix()
Y = dfv.get_whole_class_matrix()

# Filter ngrams
#    that are used in few documents
nterms = X.shape[1]
ndocs  = X.shape[0]
good_terms = np.zeros(nterms, dtype=np.int8)
doc_min_thr = math.ceil(ndocs * min_term_usage)
doc_max_thr = math.floor(ndocs * max_term_usage)
for i in range(nterms):
    nz = np.nonzero(X[:, i])[0]
    if (len(nz) > doc_min_thr) and (len(nz) < doc_max_thr):
        good_terms[i] = 1

X = X[:, good_terms==1]







