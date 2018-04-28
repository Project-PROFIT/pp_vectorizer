import os

# from decouple import AutoConfig

from pp_vectorizer.pp_vectorizer import PPVectorizer

# CONFIG = AutoConfig()
DATA_FOLDER = os.path.relpath('./data/')

class TestPPVectorizer():
    def setUp(self):
        self.pp_vect = PPVectorizer(ngram_range=(1, 3))
        text1_path = os.path.join(DATA_FOLDER, 'text1.txt')
        with open(text1_path) as f:
            self.text1 = f.read()
        text11_path = os.path.join(DATA_FOLDER, 'text11.txt')
        with open(text11_path) as f:
            self.text11 = f.read()

    def test_ngrams(self):
        analyzer = self.pp_vect.build_analyzer()
        r = analyzer(self.text1)

    def test_no_cpts_in_text(self):
        analyzer = self.pp_vect.build_analyzer()
        r = analyzer(self.text11)

    def test_broaders(self):
        pp_vect = PPVectorizer(ngram_range=(1, 3), broader_prefix='brbr')
        analyzer = pp_vect.build_analyzer()
        r = analyzer(self.text1)
        assert any(s.startswith('brbr') for s in r)

    def test_related(self):
        pp_vect = PPVectorizer(ngram_range=(1, 3), related_prefix='relrel')
        analyzer = pp_vect.build_analyzer()
        r = analyzer(self.text1)
        assert any(s.startswith('relrel') for s in r)

    def test_no_cpt(self):
        pp_vect = PPVectorizer(ngram_range=(1, 3), use_concepts=False)
        analyzer = pp_vect.build_analyzer()
        r = analyzer(self.text1)
        assert not any(s.startswith('<') for s in r)

    def test_no_terms(self):
        pp_vect = PPVectorizer(ngram_range=(1, 3), terms=False)
        analyzer = pp_vect.build_analyzer()
        r = analyzer(self.text1)
        assert all(s.startswith('<') or
                   s.startswith('broader') or
                   s.startswith('related')
                   for s in r)


class TestClassification():
    def setUp(self):
        self.pp_vect = PPVectorizer(ngram_range=(1, 3))
        self.reegle_data = dict()
        categs_folder = os.path.join(DATA_FOLDER, 'categories')
        for var in os.walk(categs_folder):
            if var[0] == categs_folder:
                continue
            filenames = var[2]
            self.reegle_data[var[0]] = []
            for filename in filenames:
                full_filepath = os.path.join(var[0], filename)
                with open(full_filepath) as f:
                    self.reegle_data[var[0]].append(f.read())

    def test_classification(self):
        from sklearn import svm
        from sklearn.model_selection import GridSearchCV
        raw_docs = []
        y = []
        for cat, cat_raw_docs in self.reegle_data.items():
            for cat_raw_doc in cat_raw_docs:
                y.append(cat)
                raw_docs.append(cat_raw_doc)
        x = self.pp_vect.fit_transform(raw_docs)
        parameters = {'kernel': ('linear', 'rbf'), 'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}
        svc = svm.SVC()
        clf = GridSearchCV(svc, parameters)
        clf.fit(x, y)
        bs = clf.best_score_
        assert bs > 0.8
