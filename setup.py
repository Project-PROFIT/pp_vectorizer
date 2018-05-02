from distutils.core import setup

setup(
    name='pp_vectorizer',
    version='0.1dev',
    packages=['pp_vectorizer'],
    license='Apache2',
    requires=open('requirements.txt', 'r').read().split()
)
