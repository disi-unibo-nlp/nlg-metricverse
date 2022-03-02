import setuptools

setuptools.setup(
    name='nlg-metricverse',
    version='1.1',
    install_requires=[
        'wget',
        'gdown',
        'nltk',
        'numpy'
    ],
    package_dir={"": "nlgmetricverse"},
    packages=setuptools.find_packages(where="nlgmetricverse"),
    python_requires=">=3.6",
    url='https://github.com/disi-unibo-nlu/nlg-metricverse',
    license='MIT',
    author='Andrea Zammarchi',
    author_email='andrea.zammarchi3@studio.unibo.it',
    description='NLG metrics evaluation'
)
