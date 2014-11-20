'''
IO
==

Functions for saving and loading knot data from files.
'''

import json
import numpy as n


def to_json_file(points, filen):
    '''
    Writes the given points to the given filename, using
    json format.

    Parameters
    ----------
    points : array-like
        nx3 array of points to save
    filen : str
        The (relative) filename to save in
    '''
    points = n.array(points).tolist()
    with open(filen, 'w') as fileh:
        json.dump(points, fileh)

def from_json_file(filen):
    '''
    Loads an array of points from the given filename assuming
    json format, and returns the result as a numpy array.

    Parameters
    ----------
    filen : str
        The (relative) filename to load from
    '''
    with open(filen, 'r') as fileh:
        points = json.load(fileh)

    return n.array(points)


def from_csv(filen, index_col=None, header_row=None, **kwargs):
    '''
    Loads an array of points from the given filename parsed
    as a csv.
    
    .. note:: This function requires pandas to be installed.

    .. note:: For data in a space-separated column format,
              pass the argument `sep=' '` to read in as a csv.

    Parameters
    ----------
    filen : str
        The (relative) filename to load from.
    index_col: int or None
        The column that indexes the rows, defaults to None
        meaning no such column is present.
    header_row: int or None
        The row with header information (i.e. column names), with
        all previous rows ignored. Defaults to None, which means
        no such header exists.

    **kwargs :
        Passed directly to pandas.DataFrame.from_csv.
    '''
    import pandas as pn
    df = pn.DataFrame.from_csv(filen, index_col=index_col,
                               header_row=header_row, **kwargs)
    return df.as_matrix()



