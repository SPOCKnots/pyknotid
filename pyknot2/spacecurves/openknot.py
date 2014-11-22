'''
OpenKnot
========

A class for working with the topology of open curves.
'''

import numpy as n

from pyknot2.spacecurves.spacecurve import SpaceCurve
from pyknot2.spacecurves.knot import Knot
from pyknot2.spacecurves.rotation import get_rotation_angles, rotate_to_top


class OpenKnot(SpaceCurve):
    '''
    Class for holding the vertices of a single line that is assumed to
    be an open curve. This class inherits from
    :class:`~pyknot2.spacecurves.spacecurve.SpaceCurve`, replacing any
    default arguments that assume closed curves, and providing methods
    for statistical analysis of knot invariants on projection and closure.

    All knot invariant methods return the results of a sampling over
    many projections of the knot, unless indicated otherwise.
    '''

    @property
    def points(self):
        return super(OpenKnot, self).points

    @points.setter
    def points(self, points):
        super(OpenKnot, self.__class__).points.fset(self, points)
        self._cached_alexanders = None

    def raw_crossings(self, mode='use_max_jump', 
                      recalculate=False, use_python=False):
        '''
        Calls :meth:`pyknot2.spacecurves.spacecurve.SpaceCurve.raw_crossings`,
        but without including the closing line between the last
        and first points (i.e. setting include_closure=False).
        '''
        return super(OpenKnot, self).raw_crossings(mode=mode,
                                                   include_closure=False,
                                                   recalculate=recalculate,
                                                   use_python=use_python)

    def __str__(self):
        if self._crossings is not None:
            return '<OpenKnot with {} points, {} crossings>'.format(
                len(self.points), len(self._crossings))
        return '<OpenKnot with {} points>'.format(len(self.points))

    def arclength(self):
        '''Calls :meth:`pyknot2.spacecurves.spacecurve.SpaceCurve.arclength`,
        automatically *not* including the closure.
        '''
        return super(OpenKnot, self).arclength(include_closure=False)

    def smooth(self, repeats=1, window_len=10, window='hanning'):
        '''Calls :meth:`pyknot2.spacecurves.spacecurve.SpaceCurve.smooth`,
        with the periodic argument automatically set to False.
        '''
        super(OpenKnot, self).smooth(repeats=repeats, window_len=window_len,
                                     window=window, periodic=False)

    def plot_uniform_angles(self, number_of_samples):
        '''
        Plots the projection of the knot at each of the given
        number of samples, approximately evenly distributed on
        the sphere.

        This function is really just for testing.
        '''
        angles = get_rotation_angles(number_of_samples)

        for i, angs in enumerate(angles):
            k = OpenKnot(self.points, verbose=False)
            k._apply_matrix(rotate_to_top(*angs))
            fig, ax = k.plot_projection(show=False)
            fig.set_size_inches((2, 2))
            fig.savefig('rotation{:05d}'.format(i))

    def alexander_polynomials(self, number_of_samples=10, radius=None,
                              zero_centroid=False):
        '''
        Returns a list of Alexander polynomials for the knot, closing
        on a sphere of the given radius, with the given number of sample
        points approximately evenly distributed on the sphere.

        The results are cached by number of samples and radius.

        Parameters
        ----------
        number_of_samples : int
            The number of points on the sphere to sample. Defaults to 10.
        radius : float
            The radius of the sphere on which to close the knot. Defaults
            to None, which picks 10 times the largest Cartesian deviation
            from 0.
        zero_centroid : bool
            Whether to first move the average position of vertices to
            (0, 0, 0). Defaults to False.

        Returns
        -------
        : ndarray
            A number_of_samples by 3 array of angles and alexander
            polynomials.
        '''
        if zero_centroid:
            self.zero_centroid()
        
        if self._cached_alexanders is not None:
            if (number_of_samples, radius) in self._cached_alexanders:
                return self._cached_alexanders[(number_of_samples,
                                                radius)]
        else:
            self._cached_alexanders = {}
        angles = get_rotation_angles(number_of_samples)

        polys = []

        cache_radius = radius

        if radius is None:
            radius = 10*n.max(self.points)
            # Not guaranteed to give 10* the real radius, but good enough

        for i, angs in enumerate(angles):
            k = Knot(self.points, verbose=False)
            k._apply_matrix(rotate_to_top(*angs))
            if zero_centroid:
                k.zero_centroid()
            points = k.points
            closure_point = points[-1] + points[0] / 2.
            closure_point[2] = radius
            k.points = n.vstack([points, closure_point])

            polys.append([angs[0], angs[1], k.alexander_polynomial()])

        self._cached_alexanders[
            (number_of_samples, cache_radius)] = n.array(polys)

        return n.array(polys)

    def alexander_fractions(self, number_of_samples=10, **kwargs):
        '''Returns each of the Alexander polynomials from
        self.alexander_polynomials, with the fraction of that type.
        '''
        polys = self.alexander_polynomials(
            number_of_samples=number_of_samples, **kwargs)
        alexs = n.round(polys[:, 2]).astype(n.int)

        fracs = []
        length = float(len(alexs))
        for alex in n.unique(alexs):
            fracs.append((alex, n.sum(alexs == alex) / length))
        #fracs = n.array(fracs)

        return sorted(fracs, key=lambda j: j[1])
        #return fracs[n.argsort(fracs[:, 1])]

    def _alexander_map_values(self, number_of_samples=10, **kwargs):
        polys = self.alexander_polynomials(
            number_of_samples=number_of_samples, **kwargs)

        from scipy.interpolate import griddata

        positions = []
        for i, row in enumerate(polys):
            positions.append(gall_peters(row[0], row[1]))
        positions = n.array(positions)

        interpolation_points = n.mgrid[0:2*n.pi:157j,
                                       -2.:2.:100j]
        values = griddata(positions, polys[:, 2],
                          tuple(interpolation_points),
                          method='nearest')

        return positions, values
        

    def plot_alexander_map(self, number_of_samples=10,
                           scatter_points=False,
                           mode='imshow', **kwargs):
        '''
        Creates (and returns) a projective diagram showing each
        different Alexander polynomial in a different colour according
        to a closure on a far away point in this direction.
        '''

        positions, values = self._alexander_map_values(number_of_samples)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        if mode == 'imshow':
            ax.imshow(values.T, cmap='rainbow', interpolation='none')
        else:
            ax.contourf(values.T, cmap='rainbow',
                        levels=[0] + range(3, int(n.max(values)+1.1), 2))
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_xlim(-0.5, 156.5)
        ax.set_ylim(-0.5, 99.5)

        im_positions = positions*25
        im_positions[:, 0] -= 0.5
        im_positions[:, 1] += 49.5
        print im_positions
        if scatter_points:
            ax.scatter(im_positions[:, 0], im_positions[:, 1], color='black',
                       alpha=1, s=1)
        
        fig.tight_layout()
        fig.show()

        return fig, ax

    def plot_alexander_shell(self, number_of_samples=10, **kwargs):

        positions, values = self._alexander_map_values(number_of_samples, **kwargs)

        thetas = n.arcsin(n.linspace(-1, 1, 100)) + n.pi/2.
        phis = n.linspace(0, 2*n.pi, 157)

        thetas, phis = n.meshgrid(thetas, phis)

        r = 10.
        zs = r*n.cos(thetas)
        xs = r*n.sin(thetas)*n.cos(phis)
        ys = r*n.sin(thetas)*n.sin(phis)

        import mayavi.mlab as may
        print xs.shape, ys.shape, zs.shape, values.shape
        may.mesh(xs, ys, zs, scalars=values)
                                   
                             
                          

def gall_peters(theta, phi):
    '''
    Converts spherical coordinates to the Gall-Peters
    projection of the sphere, an area-preserving projection in
    the shape of a Rectangle.

    Parameters
    ----------
    theta : float
        The latitude, in radians.
    phi : float
        The longitude, in radians.
    '''
    theta -= n.pi/2
    return (phi, 2*n.sin(theta))
