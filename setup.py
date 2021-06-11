#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="basepairmodels",
    version='0.2.0',
    description=("BPNet: toolkit to learn motif synthax from high-resolution functional genomics data"
                 " using convolutional neural networks"),
    author="Zahoor Zafrulla",
    author_email="zahoor@stanford.edu",
    url="https://github.com/kundajelab/basepairmodels",
    packages=find_packages(exclude=["docs", "docs-build"]),
    install_requires=["tensorflow-gpu==2.4.1", "tensorflow-probability", "tqdm",
                      "scikit-learn", "scipy", "scikit-image", "scikit-learn", 
                      "numpy", "deepdish", "pandas", "matplotlib", "plotly", 
                      "deeptools", "pyfaidx", "modisco==0.5.14.1", "deeplift", 
                      "shap @ git+https://github.com/AvantiShri/shap.git", 
                      "mseqgen @ git+https://github.com/juanelenter/mseqgen.git"],
    extras_require={"dev": ["pytest", "pytest-cov"]},
    license="MIT license",
    zip_safe=False,
    keywords=["deep learning",
              "computational biology",
              "bioinformatics",
              "genomics"],
    test_suite="tests",
    include_package_data=True,
    tests_require=["pytest", "pytest-cov"],
    entry_points = {
        "console_scripts": [
            "train = basepairmodels.cli.bpnettrainer:main",
            "predict = basepairmodels.cli.predict:predict_main",
            "metrics = basepairmodels.cli.metrics:metrics_main",
            "interpret = basepairmodels.cli.interpret:interpret_main",
            "modisco = basepairmodels.cli.run_modisco:modisco_main",
            "logits2profile = basepairmodels.cli.logits2profile:logits2profile_main",
            "bounds = basepairmodels.cli.bounds:bounds_main",
            "counts_loss_weight = basepairmodels.cli.counts_loss_weight:counts_loss_weight_main",
            "embeddings = basepairmodels.cli.embeddings:embeddings_main",            
            "shap_scores = basepairmodels.cli.shap_scores:shap_scores_main",
            "motif_discovery = basepairmodels.cli.motif_discovery:motif_discovery_main"
#            "fastpredict = basepairmodels.cli.fastpredict:predict_main"
        ]
    }
)
