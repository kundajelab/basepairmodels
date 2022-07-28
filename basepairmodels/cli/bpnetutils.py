"""
    Miscellaneous utility functions
     
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

import random
import string
import pytz

from datetime import datetime, timezone

def getAlphaNumericTag(length):
    """ generate a random alpha numeric tag
    
        Args:
            length (int): the desired length of the tag 
        
        Returns:
            str: alphanumeric string of length `length`
    
    """

    return ''.join(random.choices(string.ascii_letters + string.digits, 
                                  k=length))


def utc_to_local(utc, tz_str):
    """ convert utc time to a given timezone
        
        Args:
            utc (datetime.datetime): utc time
            tz_str (str): timezone string e.g. 'US/Pacific'
                
        Returns:
            datetime.datetime
    """
    
    # get time zone object from string
    tz = pytz.timezone(tz_str)
    
    return utc.replace(tzinfo=timezone.utc).astimezone(tz=tz)


def local_datetime_str(tz_str):
    """ string representation of local date & time
    
        Args:
            tz_str (str): timezone string e.g. 'US/Pacific'
            
        Returns:
            str
    """
    
    # get local datetime.datetime
    dt = utc_to_local(datetime.utcnow(), tz_str)
    
    # convert datetime.datetime to str
    return dt.strftime('%Y-%m-%d_%H_%M_%S')
