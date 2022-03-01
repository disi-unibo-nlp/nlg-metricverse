import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="blanche-metrics-mavagnano",
    version="0.0.1",
    author="Marco Avagnano",
    author_email="marco.avagnano@studio.unibo.it",
    description="This library includes the most important generated text evaluation metrics ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/marcoavagnano98/nlg-metrics",
    project_urls={
        "Bug Tracker": "https://github.com/marcoavagnano98/nlg-metrics",
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
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
