"""
    Custom logger for the basepairmodels library
    
    License:
    
    MIT License

    Copyright (c) 2022 Kundaje Lab

    Permission is hereby granted, free of charge, to any person 
    obtaining a copy of this software and associated documentation
    files (the "Software"), to deal in the Software without 
    restriction, including without limitation the rights to use, copy,
    modify, merge, publish, distribute, sublicense, and/or sell copies
    of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be 
    included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
    OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
    BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
    ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN 
    CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

"""

import logging
import os
import sys

def init_logger(logfname=None):
    """
        Function to setup all the logging handlers with the desired 
        level and message format
    
        Args:
            logfname (str): path to file to store logs
            
    """
    
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


    if logfname is not None:
        # create file handler which logs even debug messages
        formatter = logging.Formatter("%(levelname)s:%(asctime)s:"
                                      "[%(filename)s:%(lineno)s -"
                                      "%(funcName)20s() ] %(message)s")

        fh = logging.FileHandler(logfname)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        root.addHandler(fh)
