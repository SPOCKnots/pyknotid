'''
Building a knot database
========================

Functions for building a knot database from raw data
files. Intended for use with the RDF format data from the Knot
Atlas.'''

import sympy as sym
import rdflib
from rdflib import URIRef, Graph

import database as db
from database import Knot
import converters

rdfalex = URIRef('invariant:Alexander_Polynomial')
rdfjones = URIRef('invariant:Jones_Polynomial')
rdfhomfly = URIRef('invariant:HOMFLYPT_Polynomial')

rdfcross = URIRef('invariant:Crossings')
rdfdet = URIRef('invariant:Determinant')
rdfsig = URIRef('invariant:Signature')

rdfunknot = URIRef('invariant:Unknotting_Number')
rdfhyperbolic = URIRef('invariant:HyperbolicVolume')

rdfgc = URIRef('invariant:Gauss_Code')
rdfconway = URIRef('invariant:ConwayNotation')
rdfpd = URIRef('invariant:PD_Presentation')
rdfdtcode = URIRef('invariant:DT_Code')

rdfv2 = URIRef('invariant:V_2')
rdfv3 = URIRef('invariant:V_3')

rdfsymmetry = URIRef('invariant:Symmetry_Type')

# RDFLib arrangement is subject/predicate/object

db.db.get_conn()

# Try to create a Knot table in the database, just in case one doesn't
# exist. This might be a silly way to deal with things, but it'll do
# for now.
if not db.Knot.table_exists():
    db.Knot.create_table()


def add_knots_from(filen):
    '''Read the RDF file at filen, and add all its knots to the database
    specified in database.py.

    The filen *must* be formatted in rdf nt format. This is the case
    for knot atlas take home database files.

    '''
    g = Graph()
    g.parse(filen, format='nt')

    subjects = list(set(g.subjects(None, None)))
    # Using the set removes duplicates

    knots = []
    total = len(subjects)
    i = 0
    for subject in subjects:
        i += 1
        if i % 10 == 0:
            sys.stdout.write('\r{}: knot {} of {}'.format(filen, i, total))
            sys.stdout.flush()

        # Each subject is a knot
        identifier = str(subject.toPython().split(':')[1])

        alexander = get_rdf_object(g, subject, rdfalex)
        if alexander is not None:
            alexander = converters.rdf_poly_to_sympy(alexander, 't')
            alexander = converters.alexander_to_json(alexander)

        jones = get_rdf_object(g, subject, rdfjones)
        if jones is not None:
            jones = converters.rdf_poly_to_sympy(jones, 'q')
            jones = converters.jones_to_json(jones)

        homfly = get_rdf_object(g, subject, rdfhomfly)
        #print 'homfly is', homfly
        if homfly is not None and 'Failed' not in homfly:
            homfly = homfly.replace('\\text{QuantumGroups$\\grave{ }$', '')
            homfly = homfly.replace('a}', 'a')
            homfly = converters.rdf_poly_to_sympy(homfly, ['z','a'])
            homfly = converters.homfly_to_json(homfly)

        crossings = get_rdf_object(g, subject, rdfcross)
        if crossings is not None:
            crossings = int(crossings[0])
        else:
            crossings = int(identifier.split('_')[0][1:3])

        determinant = get_rdf_object(g, subject, rdfdet)
        if determinant is not None:
            determinant = int(determinant)

        signature = get_rdf_object(g, subject, rdfsig)
        if signature is not None:
            signature = int(signature)

        unknot_number = get_rdf_object(g, subject, rdfunknot)
        if unknot_number is not None:
            if 'math' in unknot_number:
                unknot_number = None
            else:
                unknot_number = int(unknot_number)

        hyp_vol = get_rdf_object(g, subject, rdfhyperbolic)
        if hyp_vol is not None:
            hyp_vol = str(hyp_vol)

        gauss_code = get_rdf_object(g, subject, rdfgc)
        if gauss_code is not None:
            gauss_code = str(gauss_code)

        conway = get_rdf_object(g, subject, rdfconway)
        if conway is not None:
            conway = str(conway)
            conway = conway.replace('<nowiki>', '')
            conway = conway.replace('</nowiki>', '')

        planar_diagram = get_rdf_object(g, subject, rdfpd)
        if planar_diagram is not None:
            planar_diagram = str(planar_diagram)
            planar_diagram = planar_diagram.replace('<sub>', '_')
            planar_diagram = planar_diagram.replace('</sub>', '')

        dt_code = get_rdf_object(g, subject, rdfdtcode)
        if dt_code is not None:
            dt_code = str(dt_code)

        v2 = get_rdf_object(g, subject, rdfv2)
        if v2 is not None:
            v2 = int(v2)

        v3 = get_rdf_object(g, subject, rdfv3)
        if v3 is not None:
            v3 = int(v3)

        symmetry = get_rdf_object(g, subject, rdfsymmetry)
        if symmetry is not None:
            symmetry = symmetry.lower()
        

        k = Knot(identifier=identifier,
                 min_crossings=crossings,
                 determinant=determinant,
                 signature=signature,
                 alexander=alexander,
                 jones=jones,
                 homfly=homfly,
                 unknotting_number=unknot_number,
                 hyperbolic_volume=hyp_vol,
                 conway_notation=conway,
                 gauss_code=gauss_code,
                 planar_diagram=planar_diagram,
                 dt_code=dt_code,
                 vassiliev_2=v2,
                 vassiliev_3=v3,
                 symmetry=symmetry,
                 )
        knots.append(k)
    sys.stdout.write('\n')
    sys.stdout.flush()

    print 'Attempting to save in transaction...'
    with db.db.transaction():
        for knot in knots:
            knot.save()
    return knots


def get_rdf_object(graph, subject, predicate):
    '''Takes an rdflib Graph, subject and predicate, and returns the first
    matching object if one exists. If none exist, returns None.'''
    objects = list(graph.objects(subject, predicate))
    if len(objects) == 0:
        return None
    else:
        return objects[0].toPython()

if __name__ == '__main__':
    import sys

    filens = sys.argv[1:]
    for filen in filens:
        print 'Reading in from', filen
        add_knots_from(filen)
