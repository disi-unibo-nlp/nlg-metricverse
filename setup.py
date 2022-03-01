from distutils.core import setup
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='nlg-eval',
    version='1.1',
    author='Andrea Zammarchi',
    author_email='andrea.zammarchi3@studio.unibo.it',
    description='NLG evaluation',
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    url='https://github.com/disi-unibo-nlu/nlg-eval',
    project_urls={
        "Bug Tracker": "https://github.com/disi-unibo-nlu/nlg-eval",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'wget',
        'gdown',
        'nltk'
    ],
)
