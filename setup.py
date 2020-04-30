#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="basepairmodels",
    version='0.0.1',
    description=("BPNet: toolkit to learn motif synthax from high-resolution functional genomics data"
                 " using convolutional neural networks"),
    author="Zahoor Zafrulla",
    author_email="zahoor@stanford.edu",
    url="https://github.com/kundajelab/basepairmodels",
    packages=find_packages(),
    install_requires=["tensorflow-gpu==1.14", "keras==2.2.4", "scikit-learn", 
                      "scipy", "scikit-image", "scikit-learn", "deepdish", 
                      "h5py", "numpy", "pandas", "matplotlib", "plotly", 
                      "deeptools", "pyfaidx", "modisco", "deeplift", "tqdm",
                      "shap @ git+https://github.com/AvantiShri/shap.git"],
    extras_require={"dev": ["pytest", "pytest-cov"]},
    license="MIT license",
    zip_safe=False,
    keywords=["deep learning",
              "computational biology",
              "bioinformatics",
              "genomics"],
    test_suite="tests",
    include_package_data=True,
    tests_require=["pytest", "pytest-cov"]
)
