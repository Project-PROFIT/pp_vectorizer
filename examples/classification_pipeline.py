from pp_vectorizer import pp_vectorizer as ppv
from sklearn.neural_network import MLPClassifier
from pp_vectorizer.doc_organizer import MultilableDocOrganizer
from pp_vectorizer.doc_organizer import MultilabelDocClassificationPipeline \
    as mldcp
import pp_vectorizer.file_utils as fu

import os
import numpy as np
import math

# --- Parameters
base_folder = os.getenv("DOCS_PATH")
vectorizer_parameters = {'ngram_range': (1, 3),
                         'max_df': 0.45,
                         'max_features': 1000
                         }
classifier_parameters = {'max_iter': 500}

dfv = MultilableDocOrganizer(base_folder)
all_locations = dfv.get_locations()
all_classes = dfv.get_class_matrix()
class_names = dfv.get_category_names()



numtrain = int(0.8 * len(all_locations))
locs_train = all_locations[:numtrain]
y_train = all_classes[:numtrain, :]
locs_test = all_locations[numtrain+1:]
y_test = all_classes[numtrain+1:, :]

pipeline = mldcp(ppv.PPVectorizer,
                 MLPClassifier,
                 vectorizer_parameters,
                 classifier_parameters)
pipeline.fit(locs_train, y_train)
y_pred = pipeline.predict(locs_test)


