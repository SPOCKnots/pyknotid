'''Contains functions for improving an existing database by filling in
invariants. Original motivation was/is to get Jones polynomials from
the HOMFLY polynomials provided in the high-crossing data.'''

from database import Knot, db
from converters import homfly_to_jones, db2py_homfly, py2db_jones

import csv
import json
import numpy as n

def jones_from_homfly():
    '''Takes any knots with a homfly polynomial but lacking jones, and
    computes the latter from the former.'''

    knots = Knot.select().where(Knot.jones >> None)

    output_knots = []
    i = 0
    for knot in knots:
        if i % 100 == 0:
            print i, len(output_knots)
        i += 1
        if knot.jones is None and knot.homfly is not None:
            homfly = knot.homfly
            if 'Fail' not in homfly:
                homfly = db2py_homfly(homfly)
                jones = homfly_to_jones(homfly)
                jones = py2db_jones(jones)
                knot.jones = jones
                output_knots.append(knot)

        if i % 10000 == 0:
            print 'Saving changes'
            with db.transaction():
                for knot in output_knots:
                    knot.save()
            output_knots = []

    print 'Saving changes'
    with db.transaction():
        for knot in output_knots:
            knot.save()

def alexander_imags_from_alexander(min_crossings=None):
    '''Takes any knots with an Alexander polynomial but lacking its values
    at exp(2 pi I / 3) and exp(2 pi I / 4). It could also do the
    determinant (some of the determinants in the knot atlas data are
    wrong), but doesn't for now.
    '''
    knots = Knot.select().where(~(Knot.alexander >> None))
    if min_crossings is not None:
        knots = knots.where(Knot.min_crossings == min_crossings)

    output_knots = []
    for i, knot in enumerate(knots):
        if i % 100 == 0:
            print i, len(output_knots)
        if knot.alexander is not None:
            array = json.loads(knot.alexander)
            exp_3s = [n.exp(i*2*n.pi*1j/3.) for i in range(len(array))]
            alexander_imag_3 = int(n.round(n.abs(n.sum([
                coefficient * exponential for coefficient, exponential
                in zip(array, exp_3s)]))))

            exp_4s = [(1.j)**i for i in range(len(array))]
            alexander_imag_4 = int(n.round(n.abs(n.sum([
                coefficient * exponential for coefficient, exponential
                in zip(array, exp_4s)]))))

            knot.alexander_imag_3 = alexander_imag_3
            knot.alexander_imag_4 = alexander_imag_4
            output_knots.append(knot)

        if i % 10000 == 0:
            print('Saving changes')
            with db.transaction():
                for knot in output_knots:
                    knot.save()
            output_knots = []
    print('Saving changes')
    with db.transaction():
        for knot in output_knots:
            knot.save()
    output_knots = []
        
            
def add_fiberedness():
    '''Adds fiberedness information to any knots where the information is
    available in knotinfo's database.'''
    data = []
    with open('/home/asandy/knotcatalogue/raw_data/knotinfo_data_complete.csv','r') as fileh:
        reader = csv.reader(fileh, delimiter=',', quotechar='"')
        for row in reader:
            entries = []
            for column in row:
                entries.append(column)
            data.append(entries)

    output_knots = []
    for row in data[2:]:
        name = row[0]
        monodromy = row[158]
        fibered = False
        if monodromy != 'Not Fibered':
            fibered = True

        crossings, number = name.split('_')
        number = int(number)
        if int(crossings[:2]) >= 11:
            realname = 'K{}{}'.format(crossings, number)
        else:
            crossings = int(crossings)
            realname = '{}_{}'.format(crossings, number)
        knots = list(Knot.select().where(Knot.identifier==realname))
        if len(knots) != 1:
            print 'Failed to find {} in db'.format(realname)
        else:
            knot = knots[0]
            knot.fibered = fibered
            output_knots.append(knot)

    print 'Attempting to save'
    with db.transaction():
        for knot in output_knots:
            knot.save()
        
def add_two_bridgeness():
    '''Adds two-bridge information to any knots where the information is
    available in knotinfo's database.
    '''
    data = []
    with open('/home/asandy/knotcatalogue/raw_data/knotinfo_data_complete.csv','r') as fileh:
        reader = csv.reader(fileh, delimiter=',', quotechar='"')
        for row in reader:
            entries = []
            for column in row:
                entries.append(column)
            data.append(entries)

    output_knots = []
    for row in data[2:]:
        name = row[0]
        twobridgenotation = row[24]
        if twobridgenotation == '':
            twobridgenotation = None

        crossings, number = name.split('_')
        number = int(number)
        if int(crossings[:2]) >= 11:
            realname = 'K{}{}'.format(crossings, number)
        else:
            crossings = int(crossings)
            realname = '{}_{}'.format(crossings, number)
        knots = list(Knot.select().where(Knot.identifier==realname))
        if len(knots) != 1:
            print 'Failed to find {} in db'.format(realname)
        else:
            knot = knots[0]
            knot.two_bridge = twobridgenotation
            output_knots.append(knot)

    print 'Attempting to save'
    with db.transaction():
        for knot in output_knots:
            knot.save()
        

    
        
