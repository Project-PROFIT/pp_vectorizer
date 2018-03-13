import numpy as np
import os
from .file_utils import string_hasher
from .file_utils import TextFileIterator as dociter
from sklearn.datasets.base import load_data

class DocumentGroupVectorizer:
    def __init__(self, vectorizer, superfolder=None):
        self.vectorizer = vectorizer
        self.docs = dict()
        self.hashes_locations = dict()
        self.insync = False
        self.hash_list = []
        self.tdm = None
        self.categories = []
        self.feature_names = []
        self.hash_indices = dict()
        if superfolder:
            categories = os.listdir(superfolder)
            for cat in categories:
                locations = os.listdir(superfolder + "/" + cat)
                locations = [superfolder + "/" + cat + "/" + x
                             for x in locations]
                self.add_category(cat, doc_locations=locations)


    def _hash_location(self,location):
        with open(location, "rt") as lf:
            text = lf.read()
            return string_hasher(text)
        raise FileNotFoundError

    def add_category(self, name, doc_locations=[]):
        if name not in self.docs.keys():
            self.docs[name] = []
        for x in doc_locations:
            self.add_document(name, x)
        self.categories = list(self.docs.keys())
        self.insync = False

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
        self.insync = False

    def transform(self):
        pass

    def fit_transform(self):
        """
        Vectorize all currently loaded documents
        :return: 
        """
        self.hash_list = list(self.hashes_locations.keys())
        self.hash_list.sort()
        self.hash_indices = {x:i for i,x in enumerate(self.hash_list)}
        all_locations = [self.hashes_locations[h]
                         for h in self.hash_list]
        tf_iter = dociter(all_locations)
        self.tdm = self.vectorizer.fit_transform(tf_iter)
        self.feature_names = self.vectorizer.get_feature_names()
        self.insync = True

    def get_categories(self):
        return self.docs.keys()

    def get_matrix_for_category(self, category):
        if not self.insync:
            self.fit_transform()
        X = np.zeros((len(self.docs[category]), self.tdm.shape[1] ))
        for i, ha in enumerate(self.docs[category]):
            doc_idx = self.hash_indices[ha]
            X[i, :] = self.tdm[doc_idx, :].toarray()
        return X

    def get_whole_matrix(self):
        if not self.insync:
            self.fit_transform()
        return self.tdm

    def get_whole_class_matrix(self):
        if not self.insync:
            self.fit_transform()
        Y = np.zeros((self.tdm.shape[0], len(self.categories)))
        for i, cat in enumerate(self.categories):
            for h in self.docs[cat]:
                doc_idx = self.hash_indices[h]
                Y[doc_idx, i] = 1

        return Y


