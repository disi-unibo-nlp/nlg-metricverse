# Lint as: python3
""" NLG Metricverse is an end-to-end Python library for NLG evaluation, devised to provide
a living unified codebase for fast application, analysis, comparison, visualization, and
prototyping of automatic metrics [COLING22].


The setup.py file contains information about the package that PyPi needs, like its name, a description, the current version, etc.

Note:
   VERSION needs to be formatted following the MAJOR.MINOR.PATCH convention
   (we need to follow this convention to be able to retrieve versioned scripts)
   
To create the package for pypi.

0. Prerequisites:
   - Dependencies:
     - twine: "pip install twine"
   - Create an account in (and join the 'nlg-metricverse' project):
     - PyPI: https://pypi.org/
     - Test PyPI: https://test.pypi.org/
1. Change the version in:
   - __init__.py
   - setup.py

2. Commit these changes: "git commit -m 'Release: VERSION'"

3. Add a tag in git to mark the release: "git tag VERSION -m 'Add tag VERSION for pypi'"
   Push the tag to remote: git push --tags origin master

4. Build both the sources and the wheel. Do not change anything in setup.py between
   creating the wheel and the source distribution (obviously).
   
   First, delete any "build" directory that may exist from previous builds.
   
   For the wheel, run: "python setup.py bdist_wheel" in the top level directory.
   (this will build a wheel for the python version you use to build it).
   
   For the sources, run: "python setup.py sdist"
   You should now have a /dist directory with both .whl and .tar.gz source versions.

5. Check that everything looks correct by uploading the package to the pypi test server:

   twine upload dist/* -r pypitest --repository-url=https://test.pypi.org/legacy/
   
   Check that you can install it in a virtualenv/notebook by running:
   pip install huggingface_hub fsspec aiohttp
   pip install -U tqdm
   pip install -i https://testpypi.python.org/pypi datasets
   
6. Upload the final version to actual pypi:
   twine upload dist/* -r pypi
   
7. Fill release notes in the tag in github once everything is looking hunky-dory.

8. Change the version in __init__.py and setup.py to X.X.X+1.dev0 (e.g. VERSION=1.18.3 -> 1.18.4.dev0).
   Then push the change with a message 'set dev version'
"""

import io
import os
import platform
import re

import setuptools


def get_long_description():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    with io.open(os.path.join(base_dir, "README.md"), encoding="utf-8") as f:
        return f.read()


def get_version():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(current_dir, "nlgmetricverse", "__init__.py")
    with io.open(version_file, encoding="utf-8") as f:
        return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)


def add_pywin(reqs):
    if platform.system() == "Windows":
        # Latest PyWin32 build (301) fails, required for sacrebleu
        ext_package = ["pywin32==302"]
    else:
        ext_package = []
    reqs.extend(ext_package)


"""
Mandatory requirements.
"""

requirements = [
    "datasets>=2.0.0",
    "fire>=0.4.0",
    "nltk>=3.6.6,<3.7.1",
    "numpy>=1.21.0",
    "pandas>=1.1.5",
    "rouge-score==0.1.2",
    "setuptools>=65.5.1",
    "requests>=2.27.1",
    "click==8.1.3",
    "syllables>=1.0.3",
    "typing>=3.7.4.3",
    "packaging>=21.3",
    "scipy>=1.7.3",
    "matplotlib>=3.5.1",
    "textstat>=0.7.3",
    "codecarbon==2.1.4",
    "validators>=0.20.0",
    "seaborn>=0.12.0",
    "torch>=1.12.0",
    "transformers>=4.24.0",
    "bert_score>=0.3.11",
    "tqdm>=4.64.1",
    "evaluate>=0.2.2,<=0.3",
    "pyemd>=0.5.1",
    "ipython>=7.16.1",
    "ecco>=0.1.2"
]


"""
Extra (optional) requirements. This is a feature of `pip`.
Recommended dependencies that are not required for all uses of the nlg-metricverse library.
Extra requirements are only installed as needed: they are not automatically installed unless related features
are called by the user or another package (directly or indirectly).
They can be forced by putting them in `install-requires`.
Note: they are organized like a dictionary (key: optional_feature, value: requirements) and can be installed
by running `pip install nlg-metricverse[optional_feature]`.
"""

_DEV_REQUIREMENTS = [
    "black==21.7b0",
    "deepdiff==5.5.0",
    "flake8==3.9.2",
    "isort==5.9.2",
    "pytest>=7.0.1",
    "pytest-cov>=3.0.0",
    "pytest-timeout>=2.1.0",
    "pytorch-transformers==1.2.0",
    "math_equivalence @ git+https://github.com/hendrycks/math.git",  # for datasets test metric
    "fairseq @ git+https://github.com/pytorch/fairseq.git",
    "wget==3.2"
]

_PRISM_REQUIREMENTS = ["validators"]

_METRIC_REQUIREMENTS = [
    "sacrebleu>=2.0.0",
    "jiwer>=2.3.0",
    "seqeval==1.2.2",
    "sentencepiece==0.1.96",
    "bleurt @ git+https://github.com/google-research/bleurt.git",
    "unbabel-comet>=1.1.0",
]

_METRIC_REQUIREMENTS.extend(_PRISM_REQUIREMENTS)
add_pywin(_METRIC_REQUIREMENTS)

extras = {
    "prism": _PRISM_REQUIREMENTS,
    "metrics": _METRIC_REQUIREMENTS,
    "dev": _DEV_REQUIREMENTS + _METRIC_REQUIREMENTS,
}


setuptools.setup(
    name='nlg-metricverse',
    version=get_version(),
    author='DISI UniBo NLP',
    author_email='disi.unibo.nlp@gmail.com',
    license='MIT',
    description='An End-to-End Library for Evaluating Natural Language Generation.',
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url='https://github.com/disi-unibo-nlp/nlg-metricverse',
    download_url='https://github.com/disi-unibo-nlp/nlg-metricverse/archive/refs/tags/0.9.6.tar.gz',
    packages=setuptools.find_packages(exclude=["tests"]),
    python_requires=">=3.7",
    install_requires=requirements, # dependencies NECESSARY to run the project
    extras_require=extras, # OPTIONAL dependencies (installed as needed)
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "nlgmetricverse=nlgmetricverse.cli:app",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha", # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Education",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="natural-language-processing natural-language-generation nlg-evaluation metrics language-models visualization python pytorch"
)
