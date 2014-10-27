'''
Model classes and associated functions for storing and accessing
knots in an sqlite database.

These models use the peewee ORM. Other ORMs are not currently supported!
'''

from peewee import *
import json
import os
import converters

directory = os.path.dirname(os.path.realpath(__file__)) + '/knots.db'
DB_LOCATION = directory
# The location of the database to work with.

db = SqliteDatabase(DB_LOCATION)
db.connect()


class BaseModel(Model):
    class Meta(object):
        database = db


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

    def __str__(self):
        if self.name:
            return '<Knot {} ({})>'.format(self.identifier, self.name)
        return '<Knot {}>'.format(self.identifier)

    def __repr__(self):
        return str(self)

    def pprint(self):
        '''Pretty print all information contained about self.'''
        if self.name:
            print 'Name: {}'.format(self.name)
        if self.identifier:
            print 'Identifier: {}'.format(self.identifier)
        if self.min_crossings:
            print 'Min crossings: {}'.format(self.min_crossings)
        if self.fibered is not None:
            print 'Fibered: {}'.format(self.fibered)
        if self.gauss_code:
            print 'Gauss code: {}'.format(self.gauss_code)
        if self.planar_diagram:
            print 'Planar diagram: {}'.format(self.planar_diagram)
        if self.determinant:
            print 'Determinant: {}'.format(self.determinant)
        if self.signature:
            print 'Signature: {}'.format(self.signature)
        if self.alexander:
            print 'Alexander: {}'.format(
                converters.json_to_alexander(self.alexander))
        if self.jones:
            print 'Jones: {}'.format(
                converters.json_to_jones(self.jones))
        if self.homfly:
            print 'HOMFLY: {}'.format(
                converters.json_to_homfly(self.homfly))
        if self.hyperbolic_volume:
            print 'Hyperbolic voolume: {}'.format(self.hyperbolic_volume)

