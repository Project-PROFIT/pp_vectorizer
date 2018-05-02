from distutils.core import setup

requirements = open('requirements.txt', 'r').read().split()
requirements.append("https://github.com/artreven/pp_api/tarball#egg=pp_api")

setup(
    name='pp_vectorizer',
    version='0.1dev',
    packages=['pp_vectorizer'],
    license='Apache2',
    requires=[x for x in requirements if x[0] != '-']
)
