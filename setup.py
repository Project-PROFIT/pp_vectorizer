from setuptools import setup

requirements = open('requirements.txt', 'r').read().split("\n")
requirements = [x for x in requirements if ((len(x) > 0)
                                            and (x[0] != '-')
                                            and ("+" not in x))]
requirements = [x.replace("python-", "python_") for x in requirements]
dependencies = ["https://github.com/semantic-web-company/pp_api/tarball/master#egg=pp_api"]

setup(
    name='pp_vectorizer',
    version='0.1dev',
    description='text vectorized based on PoolParty API',
    packages=['pp_vectorizer'],
    license='Apache2',
    install_requires=requirements,
    dependency_links=dependencies
)
