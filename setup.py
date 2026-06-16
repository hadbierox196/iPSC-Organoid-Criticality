"""Package installation for iPSC-Organoid-Criticality pipeline."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    requirements = [
        line.strip() for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="ipsc_organoid_criticality",
    version="1.0.0",
    author="[Author]",
    author_email="author@institution.edu",
    description=(
        "Electrophysiological criticality analysis pipeline "
        "for psychiatric iPSC organoids."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/iPSC-Organoid-Criticality",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "organoid-train=training.trainer:main",
            "organoid-eval=evaluation.evaluator:main",
            "organoid-preprocess=data.preprocessor:main",
        ]
    },
)
