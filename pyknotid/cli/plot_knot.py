#!/usr/bin/env python2

from __future__ import print_function, division

import argparse
import sys
import numpy as np

from os import path

import pyknotid.visualise as pvis

from time import sleep

from os import path
sys.path.append(path.split(path.dirname(path.realpath(__file__)))[0])

from pyknotid.spacecurves.knot import Knot

from vispy import scene


def knot_from_file(filen):
    with open(filen, 'r') as fileh:
        text = fileh.read()

    return knot_from_text(text)
    

def knot_from_text(text):

    points = []
    for line in text.split('\n'):
        values = line.split(' ')
        if len(values) == 1 and len(values[0]) == 0:
            continue
        if len(values) != 3:
            raise ValueError('Invalid text passed.')
        points.append([float(v) for v in values])

    k = Knot(points, verbose=False)
    return k


def main():
    parser = argparse.ArgumentParser(
        description='Analyse the knot type of a curve in a data file')

    parser.add_argument('filename', help='The file to load')
    parser.add_argument('--output', help='The filepath to save an image to')

    parser.add_argument('--plot', dest='plot', action='store_true')
    parser.add_argument('--no-plot', dest='plot', action='store_false')
    parser.set_defaults(plot=True)

    parser.add_argument('--azimuth', help='Angle of camera around z axis, in degreesn',
                        default=30, type=float)
    parser.add_argument('--elevation', help='Height of camera above x-y plane, in degrees',
                        default=30, type=float)
    parser.add_argument('--scale-factor', help='Camera zoom',
                        type=float)

    args = parser.parse_args(sys.argv[1:])


    k = knot_from_file(args.filename)

    k.plot()

    vispy_canvas = pvis.vispy_canvas

    vispy_canvas.view.camera = scene.TurntableCamera(fov=30)
    camera = vispy_canvas.view.camera

    if args.azimuth:
        camera.azimuth = args.azimuth
    if args.elevation:
        camera.elevation = args.elevation
    if args.scale_factor is not None:
        camera.scale_factor = args.scale_factor

    if args.output:
        pvis.vispy_save_png(args.output)

    if args.plot:
        vispy_canvas.app.run()

    camera = pvis.vispy_canvas.view.camera

    print('final camera details:')
    print('  azimuth: {}'.format(camera.azimuth))
    print('  elevation: {}'.format(camera.elevation))
    print('  scale-factor: {}'.format(camera.scale_factor))

    print(('To reuse these, add the arguments `--azimuth {} '
           '--elevation {} --scale-factor {}').format(
               camera.azimuth, camera.elevation, camera.scale_factor))
    

if __name__ == "__main__":
    main()
    
