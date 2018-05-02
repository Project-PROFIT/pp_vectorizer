from setuptools import setup


requirements = open('requirements.txt', 'r').read().split("\n")
requirements = [x for x in requirements if (len(x)>0) and  (x[0] != '-') and ("+" not in x)]
requirements = [x.replace("python-","") for x in requirements]
print(requirements)
dependencies = ["https://github.com/artreven/pp_api/tarball/master#egg=pp_api"]

setup(
    name='pp_vectorizer',
    version='0.1dev',
    description='text vectorized based on PoolParty API',
    packages=['pp_vectorizer'],
    license='Apache2',
    install_requires=requirements,
    dependency_links=dependencies
)
