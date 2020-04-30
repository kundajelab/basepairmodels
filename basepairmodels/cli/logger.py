import logging
import os
import sys

def init_logger(logfname):
    # set tensorflow logging lebel
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # ERROR
    logging.getLogger('tensorflow').setLevel(logging.ERROR)

    # remove existing handlers
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    root.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - "
                                  "%(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.flush = sys.stdout.flush
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)


    # create file handler which logs even debug messages
    formatter = logging.Formatter("%(levelname)s:%(asctime)s:"
                                  "[%(filename)s:%(lineno)s -"
                                  "%(funcName)20s() ] %(message)s")
    
    fh = logging.FileHandler(logfname)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    root.addHandler(fh)
