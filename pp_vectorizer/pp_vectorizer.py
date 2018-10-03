import logging
import re
from functools import lru_cache

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from decouple import AutoConfig
from shove import Shove

import pp_api.pp_calls as poolparty

from .doc_organizer import string_hasher

CONFIG = AutoConfig()
module_logger = logging.getLogger(__name__)


class PPCachedExtractor:
    """
    A class to provide caching capabilities for PoolParty extraction.
    """
    def __init__(self, cache_path=CONFIG('CACHE_PATH', default='simple://'),
                 store_path=CONFIG('STORE_PATH', default='simple://'),
                 pp=None):
        self.shove = Shove(store_path, cache_path)
        self.pp = pp if pp is not None else \
            poolparty.PoolParty(server=CONFIG('PP_SERVER'))

    def extract_cpts(self, text, pp_pid=CONFIG('PP_PID')):
        cache_key = string_hasher(text)
        full_key = '_'.join([cache_key, pp_pid])
        try:
            ans = self.shove[full_key]
            return ans
        except (KeyError, TypeError):
            r = self.pp.extract(text, pid=pp_pid)
            cpts = self.pp.get_cpts_from_response(r)
            self.shove[full_key] = cpts
            self.shove.sync()
            return cpts


class PPVectorizer(TfidfVectorizer):
    """
    The class vectorizes text input using PoolParty eXtractor. Additional
    features are: extracted concepts, their broaders, their related.
    """
    def __init__(self,
                 use_concepts=True,
                 use_terms=True,
                 broader_prefix='broader ',
                 related_prefix='related ',
                 cache_path=CONFIG('CACHE_PATH', default='simple://'),
                 store_path=CONFIG('STORE_PATH', default='simple://'),
                 pp_pid=None, pp=None,
                 input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None, lowercase=True,
                 preprocessor=None, tokenizer=None, analyzer='word',
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), max_df=1.0, min_df=1,
                 max_features=None, vocabulary=None, binary=False,
                 dtype=np.int64, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False
                 ):
        """
        Initialize the PoolParty Vectorizer.
        Inherits from `scikit-learn.feature_extraction.text.TfidfVectorizer`.
        Only specific parameters are described. The rest should be looked up
        in the parent class.
        Makes calls to a PoolParty instance to extract concepts and other
        metadata from documents. The class `PPCachedExtractor` is used to
        cache the results of the calls to PoolParty.

        :param use_concepts: Bool
            Default: `True`
        :param use_terms: Bool.
            Default: `True`
        :param broader_prefix: `str`
            prefix to be used with broaders.
            Provide `None` or `False` to not extract broaders.
            Default: `broader `
        :param related_prefix: `str`
            prefix to be used with relateds.
            Provide `None` or `False` to not extract relateds.
            Default: `related `
        :param cache_path: `str`
            Absolute path to the location of cache. Default: read from
            env variable *CACHE_PATH*
        :param store_path: 'str'
            Absolute path to the location of store. Default: read from
            env variable *STORE_PATH*
        :param pp_pid: str
            PoolParty Project ID. Default: read from env variable *PP_PID*
        :param pp: `PoolParty instance`
            Default: read from env variable *PP_SERVER* and create an instance.
        """
        # TODO: shadow concepts?
        super().__init__(input=input, encoding=encoding,
                         decode_error=decode_error, strip_accents=strip_accents,
                         lowercase=lowercase, preprocessor=preprocessor,
                         tokenizer=tokenizer, analyzer=analyzer,
                         stop_words=stop_words, token_pattern=token_pattern,
                         ngram_range=ngram_range, max_df=max_df, min_df=min_df,
                         max_features=max_features, vocabulary=vocabulary,
                         binary=binary, dtype=dtype, norm=norm, use_idf=use_idf,
                         smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)
        # super().__init__(*args, **kwargs)
        self.use_concepts = use_concepts
        self.broader_prefix = broader_prefix
        self.related_prefix = related_prefix
        self.use_terms = use_terms
        self.cache_path = cache_path
        self.store_path = store_path
        self.pp = pp if pp is not None else \
            poolparty.PoolParty(server=CONFIG('PP_SERVER'))
        self.pp_pid = pp_pid if pp_pid is not None else CONFIG('PP_PID')

    def build_analyzer(self):
        self.cached_extractor = PPCachedExtractor(store_path=self.store_path,
                                                  cache_path=self.cache_path,
                                                  pp=self.pp)
        self.use_broaders = isinstance(self.broader_prefix, str)
        self.use_related = isinstance(self.related_prefix, str)
        self.make_extraction = (self.use_concepts
                                or self.use_broaders
                                or self.use_related)
        """Return a callable that handles preprocessing and tokenization"""
        def analyzer(doc):
            decoded_doc = self.decode(doc).replace('<', '').replace('>', '')
            annotated_doc = decoded_doc
            result = []
            if self.make_extraction:
                module_logger.debug('Starting extraction, doc length={}'.format(
                    len(doc)))
                cpts = self.cached_extractor.extract_cpts(decoded_doc,
                                                          pp_pid=self.pp_pid)
                if self.use_concepts and self.use_concepts != 'append':
                    positions2uri = dict()
                    for cpt in cpts:
                        if 'matchings' in cpt:
                            positions2uri.update({
                                pos: '<' + cpt['uri'] + '>'
                                for x in cpt['matchings']
                                for pos in x['positions']
                            })
                    module_logger.debug('Positions of matched concepts provided:'
                                       ' {}'.format(bool(positions2uri)))
                    if not self.use_terms:
                        result = ['<' + cpt['uri'] + '>' for cpt in cpts]
                    else:
                        if positions2uri:
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
                        else:
                            annotated_doc += ' '.join(
                                ['<' + cpt['uri'] + '>' for cpt in cpts])
                if self.use_terms:
                    prepared_doc = preprocess(annotated_doc)
                    result = self._word_ngrams(tokenize(prepared_doc),
                                               stop_words)
                if self.use_concepts and self.use_concepts == 'append':
                    result += ['<' + cpt['uri'] + '>' for cpt in cpts]
                if self.use_related:
                    result += [self.related_prefix + '<' + rel_cpt + '>'
                               for cpt in cpts
                               for rel_cpt in cpt['relatedConcepts']]
                if self.use_broaders:
                    broader_results = [
                        self.broader_prefix + '<' + br_cpt + '>'
                        for cpt in cpts
                        for br_cpt in cpt['transitiveBroaderConcepts'] or
                                      cpt['transitiveBroaderTopConcepts']]
                    result += broader_results
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
