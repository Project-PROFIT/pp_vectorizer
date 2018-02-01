import os

from pp_vectorizer.pp_vectorizer import PPVectorizer

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
        print(r)

    def test_no_cpts(self):
        analyzer = self.pp_vect.build_analyzer()
        r = analyzer(self.text11)