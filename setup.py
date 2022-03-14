import io
import os
import re

import setuptools


def get_long_description():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    with io.open(os.path.join(base_dir, "README.md"), encoding="utf-8") as f:
        return f.read()


def get_requirements():
    with open("requirements.txt") as f:
        return f.read().splitlines()


def get_version():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(current_dir, "nlgmetricverse", "__init__.py")
    with io.open(version_file, encoding="utf-8") as f:
        return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)


_DEV_REQUIREMENTS = [
    "black==21.7b0",
    "deepdiff==5.5.0",
    "flake8==3.9.2",
    "isort==5.9.2",
    "pytest>=7.0.1",
    "pytest-cov>=3.0.0",
    "pytest-timeout>=2.1.0",
    "math_equivalence @ git+https://github.com/hendrycks/math.git",  # for datasets test metric
]

_PRISM_REQUIREMENTS = ["fairseq==0.9.0", "validators"]

_METRIC_REQUIREMENTS = [
    "bert_score==0.3.11",
    "bleurt @ git+https://github.com/google-research/bleurt.git",
]

extras = {
    "prism": _PRISM_REQUIREMENTS,
    "metrics": _METRIC_REQUIREMENTS,
    "dev": _DEV_REQUIREMENTS + _METRIC_REQUIREMENTS,
}


setuptools.setup(
    name='nlg-metricverse',
    version=get_version(),
    author='Andrea Zammarchi',
    license='MIT',
    description='NLG metrics evaluation',
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url='https://github.com/disi-unibo-nlu/nlg-metricverse',
    packages=setuptools.find_packages(exclude=["tests"]),
    python_requires=">=3.7",
    install_requires=get_requirements(),
    extras_require=extras,
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "nlgmetricverse=nlgmetricverse.cli:app",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
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
    keywords="machine-learning, deep-learning, ml, pytorch, NLP, evaluation, question-answering, question-generation"
)
