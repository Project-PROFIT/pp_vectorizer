from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import cross_val_score

from pp_vectorizer import pp_vectorizer as ppv
from pp_vectorizer.doc_organizer import MultilabelDocOrganizer
from pp_vectorizer.file_utils import TextFileIterator as dociter
import os


# --- Parameters
base_folder = os.getenv("DOCS_PATH")
vectorizer_parameters = {'ngram_range': (1, 3),
                         'max_df': 0.45,
                         'max_features': 1000
                         }
classifier_parameters = {'max_iter': 500}
evaluation_score = 'f1_micro'
classifier_class = SVC
n_jobs = 2  # -2: All but one CPU
num_folds = 5

# Read the data
doc_organizer = MultilabelDocOrganizer(base_folder)
all_locations = doc_organizer.get_locations()
all_classes = doc_organizer.get_category_matrix()
class_names = doc_organizer.get_category_names()
doc_iterator = dociter(all_locations)

single_classifier = classifier_class(**classifier_parameters)
vectorizer = ppv.PPVectorizer(**vectorizer_parameters)
multi_classifier = MultiOutputClassifier(single_classifier, n_jobs=n_jobs)
pipe = Pipeline(memory=None,
                steps=[('vectorization', vectorizer),
                       ('classifier', multi_classifier)]
                )

score = cross_val_score(pipe, all_locations, all_classes,
                        cv=num_folds,
                        scoring=evaluation_score)
print(score)