import logging
import os

# from decouple import AutoConfig
from pp_vectorizer.doc_organizer import MultilabelDocOrganizer
# from pp_vectorizer.file_utils import TextFileIterator
from pp_vectorizer.pp_vectorizer import PPVectorizer

# CONFIG = AutoConfig()
DATA_FOLDER = os.path.relpath('./data/')

logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


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
        assert any(s.startswith('brbr') for s in r), '\n'.join(r)

    def test_related(self):
        pp_vect = PPVectorizer(ngram_range=(1, 3), related_prefix='relrel')
        analyzer = pp_vect.build_analyzer()
        r = analyzer(self.text1)
        assert any(s.startswith('relrel') for s in r), '\n'.join(r)

    def test_no_cpt(self):
        pp_vect = PPVectorizer(ngram_range=(1, 3), use_concepts=False)
        analyzer = pp_vect.build_analyzer()
        r = analyzer(self.text1)
        assert not any(s.startswith('<') for s in r)

    def test_no_terms(self):
        pp_vect = PPVectorizer(ngram_range=(1, 3), use_terms=False)
        analyzer = pp_vect.build_analyzer()
        r = analyzer(self.text1)
        assert all(s.startswith('<') or
                   s.startswith('broader') or
                   s.startswith('related')
                   for s in r)


class TestReegleClassification():
    def setUp(self):
        self.pp_vect = PPVectorizer(use_terms=False, ngram_range=(1, 3),
                                    use_concepts=True,
                                    broader_prefix=None,
                                    related_prefix=None
                                    )
        categs_folder = os.path.join(DATA_FOLDER, 'categories')
        doc_organizer = MultilabelDocOrganizer(categs_folder)
        self.classes = doc_organizer.get_category_names()
        self.all_classes = doc_organizer.get_category_array()
        text_iter = doc_organizer.get_text_iterator()
        self.doc_list = [x for x in text_iter]

    def test_svm_classification(self):
        from sklearn import svm
        from sklearn.model_selection import GridSearchCV

        x = self.pp_vect.fit_transform(self.doc_list)
        parameters = {'kernel': ('linear', 'rbf'),
                      'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}
        svc = svm.SVC()
        clf = GridSearchCV(svc, parameters)
        clf.fit(X=x, y=self.all_classes)
        bs = clf.best_score_
        print('SVM score')
        print(bs)
        print('SVM best params')
        print(clf.best_params_)
        assert bs >= 0.8, bs

    def test_bayes_classification(self):
        from sklearn import naive_bayes
        from sklearn.model_selection import GridSearchCV

        x = self.pp_vect.fit_transform(self.doc_list)
        parameters = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}
        nb = naive_bayes.MultinomialNB()
        clf = GridSearchCV(nb, parameters)
        clf.fit(X=x, y=self.all_classes)
        bs = clf.best_score_
        print('NB score')
        print(bs)
        print('NB best params')
        print(clf.best_params_)
        assert bs >= 0.8, bs

    def test_tree_classification(self):
        from sklearn import tree
        from sklearn.model_selection import GridSearchCV

        x = self.pp_vect.fit_transform(self.doc_list)
        parameters = {'criterion': ['gini', 'entropy'],
                      'max_features': ['auto', None]}
        dt = tree.DecisionTreeClassifier()
        clf = GridSearchCV(dt, parameters)
        clf.fit(X=x, y=self.all_classes)
        bs = clf.best_score_
        print('Decision Trees score')
        print(bs)
        print('DT best params')
        print(clf.best_params_)
        assert bs >= 0.5, bs
        clf = clf.best_estimator_

        import graphviz
        # dot_data = tree.export_graphviz(clf, out_file=None)
        # graph = graphviz.Source(dot_data)
        # graph.render("iris")
        prepared_features = self.pp_vect.get_feature_names()
        prepared_features = [
            x.replace('<', '').replace('>', '') for x in prepared_features
        ]
        dot_data = tree.export_graphviz(clf, out_file=None,
                                        feature_names=prepared_features,
                                        class_names=self.classes,
                                        filled=True, rounded=True,
                                        special_characters=True)
        graph=graphviz.Source(dot_data)
        graph.render(view=True)