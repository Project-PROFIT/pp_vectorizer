"""

Simple multilabel classification example. Each directory in DOCS_PATH/categories
is assumed to contain documents from a different category


"""

import os
from time import time

from decouple import AutoConfig
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import cross_val_score

from pp_vectorizer import pp_vectorizer as ppv
from pp_vectorizer.doc_organizer import MultilabelDocOrganizer

# Two classifiers to choose from
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

CONFIG = AutoConfig()
# --- Parameters
base_folder = CONFIG("DOCS_PATH")
vectorizer_parameters = {
    'ngram_range': (1, 3),
    'max_df': 0.51,
    'max_features': 1000}
classifier_class = SVC
classifier_parameters = {'max_iter': 500}
n_jobs = 2  # -2: All but one CPU
pipeline_parameters = {
    'cv': 5,  # number of folds
    'scoring': 'f1_micro'}



# Read the data
doc_organizer = MultilabelDocOrganizer(base_folder)
all_classes = doc_organizer.get_category_matrix()
doc_iterator = doc_organizer.get_text_iterator()

# Instantiate vectorizer and classifiers
vectorizer = ppv.PPVectorizer(**vectorizer_parameters)
single_classifier = classifier_class(**classifier_parameters)
multi_classifier = MultiOutputClassifier(single_classifier, n_jobs=n_jobs)
pipe = Pipeline(memory=None,
                steps=[('vectorization', vectorizer),
                       ('classifier', multi_classifier)])
# Train and test n-fold
start = time()
score = cross_val_score(
    pipe, doc_iterator, all_classes,
    **pipeline_parameters)
print(score)
print('time taken: {:0.3f}s'.format(time() - start))
