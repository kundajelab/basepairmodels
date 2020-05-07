######################
BPNet Documentatation
######################

*************
Installation 
*************

.. role:: bash(code)
   :language: bash

:bash:`git clone https://github.com/kundajelab/basepairmodels.git`.
:bash:`cd basepairmodels`
:bash:`pip install .`


******************
API Documentation
******************

**********************
Command Line Interface
**********************

.. toctree::
   :maxdepth: 2
   :caption: Contents:


bpnettrainer
=============

.. argparse::
    :module: basepairmodels.cli.argparsers
    :func: training_argsparser
    :prog: bpnettrainer

predict
=======

.. argparse::
    :module: basepairmodels.cli.argparsers
    :func: predict_argsparser
    :prog: predict

metrics
=======

.. argparse::
    :module: basepairmodels.cli.argparsers
    :func: metrics_argsparser
    :prog: metrics

interpret
=========

.. argparse::
    :module: basepairmodels.cli.argparsers
    :func: interpret_argsparser
    :prog: interpret

run_modisco
===========

.. argparse::
    :module: basepairmodels.cli.argparsers
    :func: modisco_argsparser
    :prog: run_modisco

********
Modules
********

Loss Functions
==============
.. automodule:: basepairmodels.cli.losses
    :members:

DataLoaders
===========
.. automodule:: basepairmodels.cli.MTBatchGenerator
    :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
