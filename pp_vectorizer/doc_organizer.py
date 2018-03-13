import numpy as np
import os
import uuid

from sklearn.base import BaseEstimator

from .file_utils import string_hasher
from .file_utils import TextFileIterator as dociter
from sklearn.datasets.base import load_data


class MultilabelDocClassificationPipeline:
    def __init__(self, vectorizer_class,
                 classifier_class,
                 vectorizer_params,
                 classifier_params):
        self.vectorizer_class = vectorizer_class
        self.vectorizer_params = vectorizer_params
        self.classifier_class = classifier_class
        self.classifier_params = classifier_params
        self.x_train = None
        self.x_test = None
        self.numclasses = 0

    def fit(self, doc_locations_train, y_train):
        self.numclasses = y_train.shape[1]
        self.vectorizer = self.vectorizer_class(**self.vectorizer_params)
        self.classifier = self.classifier_class(**self.classifier_params)
        x_train = self.vectorizer.fit_transform(dociter(doc_locations_train))
        self.classifier.fit(x_train, y_train)
        self.x_train = x_train
        if hasattr(self.vectorizer, 'cache_extractor'):
            print("writing to cache")
            self.vectorizer.cache_extractor.save_cache()

    def predict(self, doc_locations_test):
        x_test = self.vectorizer.transform(doc_locations_test)
        if hasattr(self.vectorizer, 'cache_extractor'):
            self.vectorizer.cache_extractor.save_cache()
        self.x_test = x_test
        return self.classifier.predict(x_test)

    def get_params(self):
        p_dict = {'vectorizer_class': self.vectorizer_class,
                  'classifier_class': self.classifier_class,
                  'vectorizer_params': self.vectorizer_params,
                  'classifier_params': self.classifier_params}
        return p_dict

    def set_params(self, p_dict):
        self.vectorizer_class = p_dict['vectorizer_class']
        self.classifier_class = p_dict['classifier_class']
        self.vectorizer_params = p_dict['vectorizer_params']
        self.classifier_params = p_dict['classifier_params']


class MultilableDocOrganizer:
    def __init__(self, superfolder=None):
        self.docs = dict()
        self.hashes_locations = dict()
        self.indices_fixed = False
        self.hash_list = []
        self.number_of_documents = 0
        self.categories = []
        self.feature_names = []
        self.hash_indices = dict()
        self.current_revision = uuid.uuid4()
        if superfolder:
            categories = next(os.walk(superfolder))[1]
            for cat in categories:
                locations = os.listdir(superfolder + "/" + cat)
                locations = [superfolder + "/" + cat + "/" + x
                             for x in locations]
                self.add_category(cat, doc_locations=locations)

    def _hash_location(self, location):
        with open(location, "rt") as lf:
            text = lf.read()
            return string_hasher(text)
        raise FileNotFoundError

    def add_category(self, name, doc_locations=[]):
        if name not in self.docs.keys():
            self.docs[name] = []
        for x in doc_locations:
            self.add_document(name, x)
        self.indices_fixed = False
        self.current_revision = uuid.uuid4()

    def add_document(self, category, location):
        """
        Add a document location to the given category
        :param category:   a category this document belongs to
        :param location:   a location where to find this document
        :return: 
        """
        doc_hash = self._hash_location(location)
        self.hashes_locations[doc_hash] = location
        if category not in self.docs.keys():
            self.docs[category] = []
        self.docs[category].append(doc_hash)
        self.indices_fixed = False
        self.current_revision = uuid.uuid4()

    def fix_indices(self):
        self.hash_list = list(self.hashes_locations.keys())
        self.hash_list.sort()
        self.hash_indices = {x: i for i, x in enumerate(self.hash_list)}
        self.number_of_documents = len(self.hash_list)
        self.categories = list(self.docs.keys())
        self.indices_fixed = True
        self.current_revision = uuid.uuid4()

    def get_category_names(self):
        if not self.indices_fixed:
            self.fix_indices()
        return self.categories

    def get_class_matrix(self):
        if not self.indices_fixed:
            self.fix_indices()
        Y = np.zeros((self.number_of_documents, len(self.categories)))
        for i, cat in enumerate(self.categories):
            for h in self.docs[cat]:
                doc_idx = self.hash_indices[h]
                Y[doc_idx, i] = 1
        return Y

    def get_locations(self):
        if not self.indices_fixed:
            self.fix_indices()
        return [self.hashes_locations[h] for h in self.hash_list]

    # def get_matrix_for_category(self, category):
    #     if not self.indices_fixed:
    #         self.fix_indices()
    #     X = np.zeros((len(self.docs[category]), self.tdm.shape[1] ))
    #     for i, ha in enumerate(self.docs[category]):
    #         doc_idx = self.hash_indices[ha]
    #         X[i, :] = self.tdm[doc_idx, :].toarray()
    #     return X
    #
    # def get_whole_matrix(self):
    #     if not self.indices_fixed:
    #         self.fix_indices()
    #     return self.tdm
    # def transform(self):
    #     pass
    #
    # def fit_transform(self):
    #     """
    #     Vectorize all currently loaded documents
    #     :return:
    #     """
    #     self.hash_list = list(self.hashes_locations.keys())
    #     self.hash_list.sort()
    #     self.hash_indices = {x:i for i,x in enumerate(self.hash_list)}
    #     all_locations = [self.hashes_locations[h]
    #                      for h in self.hash_list]
    #     tf_iter = dociter(all_locations)
    #     self.tdm = self.vectorizer.fit_transform(tf_iter)
    #     self.feature_names = self.vectorizer.get_feature_names()
    #     self.insync = True
