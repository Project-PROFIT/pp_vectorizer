import hashlib
import os
import logging

import numpy as np
import uuid

module_logger = logging.getLogger(__name__)


class MultilabelDocOrganizer:
    """
    TODO: DocString
    """
    def __init__(self, superfolder=None):
        self.docs = dict()
        self.hash2location = dict()
        self.indices_fixed = False
        self.hash_list = []
        self.number_of_documents = 0
        self.categories = []
        # self.feature_names = []
        self.hash2index = dict()
        self.current_revision = uuid.uuid4()
        if superfolder:
            categories = next(os.walk(superfolder))[1]
            for cat in categories:
                locations = os.listdir(os.path.join(superfolder, cat))
                locations = [os.path.join(superfolder, cat, x)
                             for x in locations]
                self.add_category(cat, doc_locations=locations)
        self._fix_indices()

    def _hash_location(self, location):
        # raises FileNotFoundError
        with open(location, "rb") as lf:
            text = lf.read().decode("utf-8")
            return string_hasher(text)

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
        self.hash2location[doc_hash] = location
        if category not in self.docs.keys():
            self.docs[category] = []
        self.docs[category].append(doc_hash)
        self.indices_fixed = False
        self.current_revision = uuid.uuid4()

    def _fix_indices(self):
        self.hash_list = list(self.hash2location.keys())
        self.hash_list.sort()
        self.hash2index = {x: i for i, x in enumerate(self.hash_list)}
        self.number_of_documents = len(self.hash_list)
        self.categories = list(self.docs.keys())
        self.indices_fixed = True
        self.current_revision = uuid.uuid4()

    def get_category_names(self):
        if not self.indices_fixed:
            self._fix_indices()
        return self.categories

    def get_category_matrix(self):
        if not self.indices_fixed:
            self._fix_indices()
        Y = np.zeros((self.number_of_documents, len(self.categories)))
        for i, cat in enumerate(self.categories):
            for h in self.docs[cat]:
                doc_idx = self.hash2index[h]
                Y[doc_idx, i] = 1
        return Y

    def get_category_array(self):
        if not self.indices_fixed:
            self._fix_indices()
        Y = np.zeros(self.number_of_documents)
        for i, cat in enumerate(self.categories):
            for h in self.docs[cat]:
                doc_idx = self.hash2index[h]
                Y[doc_idx] = i
        return Y

    def get_locations(self):
        if not self.indices_fixed:
            self._fix_indices()
        return [self.hash2location[h] for h in self.hash_list]

    def get_text_iterator(self):
        """
        Iterate over all locations and read from those files.

        :return: iterator
        """
        return TextFileIterator(self.get_locations())


def string_hasher(x):
    """
    Stable hashing function. The value returned is the same regardless of time
    or system it is run on. Based on MD5
    :param x: a string to be hashed
    :return:  a string containing the hash and the length of the original,
              separated by underscore
    """
    assert isinstance(x, str)
    if len(x) == 0:
        return "EMPTY_STRING_0"
    hasher = hashlib.md5()
    hasher.update(x.encode("utf-8"))
    return str(hasher.hexdigest()) + "_" + str(len(x))


class TextFileIterator():
    """
    Iterator constructed from a list of filenames.
    In each iteration, one of said files is open as text and the contents
    returned as string
    """
    def __iter__(self):
        total_fns = len(self.filenames)
        i = 0
        for fn in self.filenames:
            i += 1
            if i % round(1+total_fns/20) == 0:
                module_logger.info('{} out of {} done'.format(i, total_fns))
            with open(fn, "rb") as tf:
                yield tf.read().decode("utf-8")

    def __init__(self, dir_or_list):
        if type(dir_or_list) == list:
            self.filenames = dir_or_list
        elif type(dir_or_list) == str:
            self.filenames = [os.path.join(dir_or_list, x)
                              for x in os.listdir(dir_or_list)
                              if os.path.isfile(os.path.join(dir_or_list, x))]
        else:
            raise TypeError(
                'The argument {} should be either list of filenames'
                ' or an absolute path as a string'.format(dir_or_list))

    def __len__(self):
        return len(self.filenames)

    def __array__(self):
        return np.array([x for x in self])

    def __getitem__(self, item):
        if item > self.__len__():
            raise IndexError
        fn = self.filenames[item]
        with open(fn, "rb") as tf:
            return tf.read().decode("utf-8")
