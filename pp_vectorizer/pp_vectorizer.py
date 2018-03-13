import os
import re
import pickle

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import pp_api.pp_calls as poolparty
from .file_utils import string_hasher


class CacheExtractor:
    def __init__(self, cache_path=os.getenv('CACHE_PATH')):
        self.cache_dict = dict()
        self.new_cache = 0
        self.cache_path = cache_path
        if os.path.exists(self.cache_path):
            with open(cache_path, 'rb') as f:
                d = pickle.load(f)
            self.cache_dict.update(d)

    def save_cache(self):
        with open(self.cache_path, 'wb') as f:
            pickle.dump(self.cache_dict, f)

    def extract(self, text, pp_pid=os.getenv('PP_PID'),
                pp=poolparty.PoolParty(server=os.getenv('PP_SERVER'))):
        cache_key = string_hasher(text)
        try:
            return self.cache_dict[(cache_key, pp_pid)]
        except KeyError:
            r = pp.extract(text, pid=pp_pid)
            self.cache_dict[(cache_key, pp_pid)] = r
            self.new_cache += 1
            if self.new_cache % 25 == 1:
                self.save_cache()
            return r


class PPVectorizer(TfidfVectorizer):
    def __init__(self,
                 use_concepts=True, terms=True,
                 broader_prefix='broader ',
                 related_prefix='related ',
                 cache_path=os.getenv('CACHE_PATH'),
                 pp_pid=os.getenv('PP_PID'),
                 pp=poolparty.PoolParty(server=os.getenv('PP_SERVER')),
                 *args, **kwargs):
        # TODO: shadow concepts?
        super().__init__(*args, **kwargs)
        self.use_concepts = use_concepts
        self.broader_prefix = broader_prefix
        self.related_prefix = related_prefix
        self.use_broaders = isinstance(broader_prefix, str)
        self.use_related = isinstance(related_prefix, str)
        self.make_extraction = (self.use_concepts
                                or self.use_broaders
                                or self.use_related)
        self.terms = terms
        self.cache_extractor = CacheExtractor(cache_path)
        self.pp = pp
        self.pp_pid = pp_pid

    def build_analyzer(self):
        """Return a callable that handles preprocessing and tokenization"""
        def analyzer(doc):
            decoded_doc = self.decode(doc).replace('<', '').replace('>', '')
            annotated_doc = decoded_doc
            result = []
            if self.make_extraction:
                extracted = self.cache_extractor.extract(decoded_doc,
                                                         pp_pid=self.pp_pid,
                                                         pp=self.pp)
                cpts = poolparty.PoolParty.get_cpts_from_response(extracted)
                if self.use_concepts:
                    positions2uri = dict()
                    for cpt in cpts:
                        if 'matchings' in cpt:
                            positions2uri.update({
                                pos: '<' + cpt['uri'] + '>'
                                for x in cpt['matchings']
                                for pos in x['positions']
                            })
                    if not self.terms:
                        result = ['<' + cpt['uri'] + '>' for cpt in cpts]
                    else:
                        sorted_pos = sorted(positions2uri.keys())
                        previous_pos = (0, -1)
                        text_fragments = []
                        for this_pos in sorted_pos:
                            text_fragments.append(decoded_doc[
                                                  previous_pos[1]+1:this_pos[0]])
                            text_fragments.append(positions2uri[this_pos])
                            previous_pos = this_pos
                        text_fragments.append(decoded_doc[previous_pos[1]+1:])
                        annotated_doc = ' '.join(text_fragments)
                if self.terms:
                    prepared_doc = preprocess(annotated_doc)
                    result = self._word_ngrams(tokenize(prepared_doc),
                                               stop_words)
                if self.use_related:
                    result += [self.related_prefix + '<' + rel_cpt + '>'
                               for cpt in cpts
                               for rel_cpt in cpt['relatedConcepts']]
                if self.use_broaders:
                    result += [self.broader_prefix + '<' + br_cpt + '>'
                               for cpt in cpts
                               for br_cpt in cpt['transitiveBroaderConcepts']]
            else:
                prepared_doc = preprocess(annotated_doc)
                result = self._word_ngrams(tokenize(prepared_doc),
                                           stop_words)
            return result

        preprocess = self.build_preprocessor()

        stop_words = self.get_stop_words()
        tokenize = self.build_tokenizer()

        return analyzer

    def build_tokenizer(self):
        token_pattern = r"(?u)<[^>]*>|\b\w\w+\b"
        token_pattern = re.compile(token_pattern)
        return lambda doc: token_pattern.findall(doc)
