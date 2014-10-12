'''
Utility functions
=================
'''

import sys

def vprint(string, newline=True, condition=True):
    '''
    Verbose print; prints the given string if the input condition
    is met.

    Pyknot2 uses this to make it easy to display changing counters.

    Parameters
    ==========
    string : str
        The string to print.
    newline : bool
        Whether to automatically append a newline.
    condition : bool
        The condition that must be met for the print to occur.
    '''
    if not condition:
        return
    sys.stdout.write(string)
    if newline:
        sys.stdout.write('\n')
    sys.stdout.flush()

def mag(v):
    '''
    Returns the magnitude of the vector v.

    Parameters
    ----------
    v : ndarray
        A vector of any dimension.
    '''
    return n.sqrt(v.dot(v))
