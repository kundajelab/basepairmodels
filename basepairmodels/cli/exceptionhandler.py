"""
    A custom class for exceptions without printing the traceback
    
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


from __future__ import print_function
import sys


def eprint(*args, **kwargs):
    """ 
        print function to print to standard error
    """
    print(*args, file=sys.stderr, **kwargs)


class NoTracebackException(Exception):
    """
        An exception that when raised results in the error message
        being printed without the traceback
    """
    pass


def notraceback_hook(kind, message, traceback):
    """
        Exception hook that reroutes all exceptions through this method
    
        Args:
            kind (type): the type of the exception
            message (obj): the exception instance
            traceback (traceback): traceback object
    """
   
    if kind.__name__ == "NoTracebackException":
        # only print message
        eprint('ERROR: {}'.format(message))  
    else:
        # print error type, message & traceback
        sys.__excepthook__(kind, message, traceback)  


# customize handling of exceptions by assigning the exception hook
sys.excepthook = notraceback_hook
