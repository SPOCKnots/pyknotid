'''
Database download module
========================

To download the database, call :func:`download_database`.

The other functions in this module provide basic functionality for
checking where the database is stored, and deleting old versions if
necessary.

API documentation
-----------------

'''

from functools import wraps
from os.path import realpath, dirname, exists, join, abspath
from os import mkdir

def find_database(db_version=None):
    '''Returns the path to the knots.db file.

    find_db looks in the following locations, in order of precedence:

    1. The local folder (containing getdb.py). This is convenient if
       you have built your own database.
    2. The directory returned by appdirs.user_data_dir (depends on the OS).

    If the database cannot be found, an exception is raised.

    You can download a prebuilt database using :func:`download_database`.

    Parameters
    ----------
    db_version : int
        The database version to find. Defaults to None, in which case the
        current db_version from :mod:`pyknotid.catalogue.database` is used.
    '''
    local_filen = join(dirname(__file__),
                       'knots.db')
    if exists(local_filen):
        return local_filen

    if db_version is None:
        from pyknotid.catalogue import db_version
    local_filen = join(dirname(realpath(__file__)),
                       'knots_{}.db'.format(db_version))
    if exists(local_filen):
        return local_filen

    import appdirs
    app_dir = appdirs.user_data_dir('pyknotid')
    app_dir_filen = join(app_dir, 'knots_{}.db'.format(db_version))
    if exists(app_dir_filen):
        return app_dir_filen

    raise IOError('Could not find a knots database file. You can '
                  'download one using '
                  '`pyknotid.catalogue.download_database()`.')


def download_target_dir():
    '''Returns the directory to which the knots database will be
    downloaded.'''
    import appdirs
    return appdirs.user_data_dir('pyknotid')

def download_database():
    '''Downloads the knots database to :func:`download_target_dir`.
    '''
    dirn = download_target_dir()
    if not exists(dirn):
        mkdir(dirn)

    from pyknotid.catalogue import db_version
    db_name = 'knots_{}.db'.format(db_version)
    filen = join(dirn, db_name)
    if exists(filen):
        raise IOError('A file named {} already exists.'.format(filen))

    import requests
    from tqdm import tqdm
    # import shutil
    r = requests.get('https://github.com/SPOCKnots/pyknotid/releases/download/init/{}'.format(db_name),
                     stream=True)

    total_size = int(r.headers.get('content-length', 0))

    with open(filen, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for data in r.iter_content(32*1024):
                pbar.update(32*1024)
                f.write(data)

    print('Successfully downloaded the new database file. Run '
          'pyknotid.catalogue.getdb.clean_old_databases to delete '
          'old database versions.')

def clean_old_databases():
    '''Deletes old database files (all but the most recent version).'''

    dirn = download_target_dir()
    import glob
    filens = glob.glob(join(dirn, 'knots_*.db'))
    versions = sorted(filens, key=lambda j: int(j.split('_')[1][:-3]))

    print('Found databases: {}'.format(', '.join(versions)))
    for version in versions[:-1]:
        print('Deleting {}'.format(version))
        print('(but not really)')

def clean_all_databases():
    '''Deletes all database files.'''

    dirn = download_target_dir()
    import glob
    filens = glob.glob(join(dirn, 'knots_*.db'))
    versions = sorted(filens, key=lambda j: int(j.split('_')[1][:-3]))

    print('Found databases: {}'.format(', '.join(versions)))
    for version in versions[:-1]:
        print('Deleting {}'.format(version))
        print('(but not really)')


def require_database(func):
    '''Decorator that causes a function to query find_database before
    returning.'''
    @wraps(func)
    def new_func(*args, **kwargs):
        find_database()
        return func(*args, **kwargs)
    return new_func
