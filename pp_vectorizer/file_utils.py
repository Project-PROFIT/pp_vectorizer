import hashlib
import os


def string_hasher(x):
    """
    Stable hashing function. The value returned is the same regardless of time
    or system it is run on. Based on MD5
    :param x: a string to be hashed 
    :return:  a string containing the hash and the length of the original,
              separated by underscore
    """
    if len(x) == 0:
        return "EMPTY_STRING_0"
    hasher = hashlib.md5()
    hasher.update(x.encode("utf-8"))
    return str(hasher.hexdigest()) + "_" + str(len(x))


class TextFileIterator:
    """
    Iterator constructed from a list of filenames.
    In each iteration, one of said files is open as text and the contents
    returned as string
    """
    def __iter__(self):
        return self

    def __init__(self, dir_or_list):
        if type(dir_or_list) == list:
            self.filenames = dir_or_list
        elif type(dir_or_list) == str:
            self.filenames = [dir_or_list + "/" + x
                              for x in os.listdir(dir_or_list)]
        else:
            raise TypeError
        self.current_file = 0

    def __next__(self):
        if self.current_file >= len(self.filenames):
            raise StopIteration
        fn = self.filenames[self.current_file]
        self.current_file += 1
        with open(fn, "rt") as tf:
            text = tf.read()
            return text
        raise FileNotFoundError



