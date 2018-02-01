import os
import re
import pickle

from sklearn.feature_extraction.text import CountVectorizer

import pp_api.pp_calls as poolparty

PP_SERVER = os.getenv('PP_SERVER')
PP_PID = os.getenv('PP_PID')
CACHE_PATH = os.getenv('CACHE_PATH')
pp = poolparty.PoolParty(server=PP_SERVER)
###
cache_dict = dict()
new_cache = 0
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH) as f:
        d = pickle.load(f)
    cache_dict.update(d)


def cache_extract(text):
    try:
        return cache_dict[text]
    except KeyError:
        r = pp.extract(text, pid=PP_PID)
        cache_dict[text] = r
        global new_cache
        new_cache += 1
        if new_cache >= 25:
            with open(CACHE_PATH, 'w') as f:
                pickle.dump(cache_dict, f)
            new_cache = 0
        return r


class PPVectorizer(CountVectorizer):
    def __init__(self,
                 extract=True, terms=True,
                 broader_prefix='broader ',
                 related_prefix='related ',
                 *args, **kwargs):
        # TODO: shadow concepts?
        super().__init__(*args, **kwargs)
        self.extract = extract
        self.terms = terms
        self.broader_prefix = broader_prefix
        self.related_prefix = related_prefix

    def build_analyzer(self):
        """Return a callable that handles preprocessing and tokenization"""
        def analyzer(doc):
            decoded_doc = self.decode(doc).replace('<', '').replace('>', '')
            annotated_doc = decoded_doc
            result = []
            if self.extract:
                r = cache_extract(decoded_doc)
                cpts = pp.get_cpts_from_response(r)
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
                result = self._word_ngrams(tokenize(prepared_doc), stop_words)
            if self.extract and self.related_prefix is not None:
                result += [self.related_prefix + '<' + rel_cpt + '>'
                           for cpt in cpts
                           for rel_cpt in cpt['relatedConcepts']]
            if self.extract and self.broader_prefix is not None:
                result += [self.broader_prefix + '<' + br_cpt + '>'
                           for cpt in cpts
                           for br_cpt in cpt['transitiveBroaderConcepts']]
            return result

        preprocess = self.build_preprocessor()

        stop_words = self.get_stop_words()
        tokenize = self.build_tokenizer()

        return analyzer

    def build_tokenizer(self):
        token_pattern = r"(?u)<[^>]*>|\b\w\w+\b"
        token_pattern = re.compile(token_pattern)
        return lambda doc: token_pattern.findall(doc)
