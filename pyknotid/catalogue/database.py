'''Database module
---------------

pyknotid looks up knot information via a prebuilt sqlite database,
accessed using the peewee ORM. Other ORMs are not currently supported.

The model class is :class:`pyknotid.catalogue.database.Knot`,
documented below. For generic documentation about using the database,
see :doc:`index`.

pyknotid also includes functions for creating a database from scratch
(using knot information from the Knot Atlas), and improving an
existing database by calculating new invariants or pulling information
from other sources such as the KnotInfo database. These functions can
be found in ``pyknotid.catalogue.build`` and
``pyknotid.catalogue.improve``, which are not included in the indexed
documentation here.

API documentation
~~~~~~~~~~~~~~~~~

'''

from peewee import *
import json
import os
from os.path import realpath, dirname, exists, join
from pyknotid.catalogue import converters

from pyknotid.catalogue.getdb import find_database

if not ('READTHEDOCS' in os.environ and os.environ['READTHEDOCS'] == 'True'):
    DB_LOCATION = find_database()
    db = SqliteDatabase(DB_LOCATION)
    db.connect()
else:
    DB_LOCATION = None
    db = None

class BaseModel(Model):
    class Meta(object):
        database = db


# If updating the db structure, don't forget to update
# pyknotid.catalogue.db_version!
class Knot(BaseModel):
    '''Peewee model for storing a knot in a database.'''

    name = CharField(null=True)
    '''The actual name (if any), e.g. trefoil'''

    identifier = CharField(null=True)
    '''The standard knot notation, e.g. 3_1 for trefoil'''

    min_crossings = IntegerField(null=True)
    '''Minimal crossing number for the knot, e.g. 3 for trefoil'''

    signature = IntegerField(null=True)
    '''The knot signature'''

    determinant = IntegerField(null=True)
    '''The knot determinant (Alexander polynomial at -1)'''

    alexander_imag_3 = IntegerField(null=True)
    '''The absolute value of the Alexander polynomial at
    exp(2 pi I / 3). This will always be an integer.'''

    alexander_imag_4 = IntegerField(null=True)
    '''The absolute value of the Alexander polynomial at
    exp(2 pi I / 4) == I. This will always be an integer.'''

    alexander = TextField(null=True)
    '''Alexander polynomial, stored as a json list of coefficients from
    lowest to highest index, including zeros if there are any jumps in
    index.
    '''

    jones = TextField(null=True)
    '''Jones polynomial, stored as a json list of coefficients and indices
    for each monomial.'''

    homfly = TextField(null=True)
    '''HOMFLY-PT polynomial, stored as a json list.'''

    unknotting_number = IntegerField(null=True)
    '''Unknotting number, stored as an integer.'''

    hyperbolic_volume = CharField(null=True)
    '''Hyperbolic volume, stored as a string to avoid precision
    problems.'''

    conway_notation = CharField(null=True)
    '''Conway notation, as a string.'''

    gauss_code = CharField(null=True)
    '''Gauss code, as a string.'''

    planar_diagram = CharField(null=True)
    '''Planar diagram representation, as a string.'''

    dt_code = CharField(null=True)
    '''Dowker-Thistlethwaite code, as a string.'''

    two_bridge = CharField(null=True)
    '''Two-bridge notation, as a string.'''

    fibered = BooleanField(null=True)
    '''Whether the knot is fibered or not.'''

    vassiliev_2 = IntegerField(null=True)
    '''The Vassiliev invariant of order 2.'''

    vassiliev_3 = IntegerField(null=True)
    '''The Vassiliev invariant of order 3.'''

    symmetry = CharField(null=True, max_length=25)
    '''The symmetry type of the knot; reversible,
    positive amphichiral, negative amphichiral fully
    amphichiral or chiral.'''

    planar_writhe = IntegerField(null=True)
    '''The writhe of the minimal diagram described by the DT_code. This is
    not necessarily unique (see Perko pair, I think?).'''

    def __str__(self):
        if self.name:
            return '<Knot {} ({})>'.format(self.identifier, self.name)
        return '<Knot {}>'.format(self.identifier)

    def __repr__(self):
        return str(self)

    def pretty_print(self):
        '''Pretty print all information contained about self.'''
        if self.name:
            print('Name: {}'.format(self.name))
        if self.identifier:
            print('Identifier: {}'.format(self.identifier))
        if self.min_crossings:
            print('Min crossings: {}'.format(self.min_crossings))
        if self.fibered is not None:
            print('Fibered: {}'.format(self.fibered))
        if self.gauss_code:
            print('Gauss code: {}'.format(self.gauss_code))
        if self.planar_diagram:
            print('Planar diagram: {}'.format(self.planar_diagram))
        if self.dt_code:
            print('DT code: {}'.format(self.dt_code))
        if self.determinant:
            print('Determinant: {}'.format(self.determinant))
        if self.signature:
            print('Signature: {}'.format(self.signature))
        if self.alexander:
            print('Alexander: {}'.format(
                converters.json_to_alexander(self.alexander)))
        if self.jones:
            print('Jones: {}'.format(
                converters.json_to_jones(self.jones)))
        if self.homfly:
            print('HOMFLY: {}'.format(
                converters.json_to_homfly(self.homfly)))
        if self.hyperbolic_volume:
            print('Hyperbolic volume: {}'.format(self.hyperbolic_volume))
        if self.vassiliev_2 is not None:
            print('Vassiliev order 2: {}'.format(self.vassiliev_2))
        if self.vassiliev_3 is not None:
            print('Vassiliev order 3: {}'.format(self.vassiliev_3))
        if self.symmetry is not None:
            print('Symmetry: {}'.format(self.symmetry))

    def space_curve(self, verbose=True, **kwargs):
        '''Returns a Knot object representing this knot.'''

        if self.dt_code is None:
            raise ValueError('No DT code is known, cannot create '
                             'space curve.')

        from pyknotid.representations import DTNotation
        import numpy as np
        d = DTNotation(self.dt_code)
        k = d.representation(verbose=verbose, **kwargs).space_curve(
            verbose=verbose)
        k.points += 0.0001 * np.random.random(k.points.shape)
        k.close()
        return k

    def url(self):
        '''The guessed url of this knot in the Knot Atlas. This page may not
        actually exist or be populated.

        '''
        return 'http://katlas.org/wiki/{}'.format(self.identifier)

    def open_url(self):
        import webbrowser
        webbrowser.open(self.url())

    def retrieve_image_path(self, **kwargs):
        images_folder = join(
            dirname(realpath(__file__)), 'diagrams_with_mirrors')

        name = self.identifier

        filen = _name_to_filen(name)

        filen = join(images_folder, filen)

        if not exists(filen):
            return None

        return filen


def _name_to_filen(name):
    if name.startswith('K'):
        name = name[1:]
        if int(name[:2]) == 11:
            digits = 3
        else:
            digits = 4
        filen = '{{}}_{{:0{}}}.png'.format(digits).format(name[:3], int(name[3:]))

    else:
        filen = name + '.png'

    return filen

