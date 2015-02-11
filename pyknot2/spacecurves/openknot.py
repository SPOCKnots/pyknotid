'''
OpenKnot
========

A class for working with the topology of open curves.
'''

from __future__ import print_function
import numpy as n

from pyknot2.spacecurves.spacecurve import SpaceCurve
from pyknot2.spacecurves.knot import Knot
from pyknot2.spacecurves.rotation import get_rotation_angles, rotate_to_top

from pyknot2.invariants import alexander
from pyknot2.representations.gausscode import GaussCode


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
                      recalculate=False, try_cython=False):
        '''
        Calls :meth:`pyknot2.spacecurves.spacecurve.SpaceCurve.raw_crossings`,
        but without including the closing line between the last
        and first points (i.e. setting include_closure=False).
        '''
        return super(OpenKnot, self).raw_crossings(mode=mode,
                                                   include_closure=False,
                                                   recalculate=recalculate,
                                                   try_cython=try_cython)

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

    def _plot_uniform_angles(self, number_of_samples):
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
            fig.close()

    def plot_projections(self, number_of_samples):
        '''
        Plots the projection of the knot at each of the given
        number of samples squared, rotated such that the sample
        direction is vertical.

        The output (and return) is a matplotlib plot with
        number_of_samples x number_of_samples axes.
        '''
        angles = get_rotation_angles(number_of_samples**2)

        print('Got angles')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(nrows=number_of_samples,
                                 ncols=number_of_samples)

        print('got axes')

        all_axes = [ax for row in axes for ax in row]

        for i, angs in enumerate(angles):
            self._vprint('i = {} / {}'.format(i, len(angles)))
            k = OpenKnot(self.points, verbose=False)
            k._apply_matrix(rotate_to_top(*angs))
            ax = all_axes[i]
            fig, ax = k.plot_projection(fig_ax=(fig, ax), show=False)

        fig.tight_layout()
        fig.show()
        return fig, ax

    def alexander_polynomials(self, number_of_samples=10, radius=None,
                              recalculate=False,
                              zero_centroid=False,
                              optimise_closure=True):
        '''
        Returns a list of Alexander polynomials for the knot, closing
        on a sphere of the given radius, with the given number of sample
        points approximately evenly distributed on the sphere.

        The results are cached by number of samples and radius.

        Parameters
        ----------
        number_of_samples : int
            The number of points on the sphere to sample. Defaults to 10.
        optimise_closure: bool
            If True, doesn't really close on a sphere but at infinity.
            This lets the calculation be optimised slightly, and so is the
            default.
        radius : float
            The radius of the sphere on which to close the knot. Defaults
            to None, which picks 10 times the largest Cartesian deviation
            from 0. This is *only* used if optimise_closure=False.
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

        if not recalculate and self._cached_alexanders is not None:
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

        print_dist = int(max(1, 3000. / len(self.points)))
        for i, angs in enumerate(angles):
            if i % print_dist == 0:
                self._vprint('\ri = {} / {}'.format(i, len(angles)), False)
            k = Knot(self.points, verbose=False)
            k._apply_matrix(rotate_to_top(*angs))
            if zero_centroid:
                k.zero_centroid()
            if optimise_closure:
                cs = k.raw_crossings()
                if len(cs) > 0:
                    closure_cs = n.argwhere(((cs[:, 0] > len(self.points)-1) & (cs[:, 2] < 0.)) |
                                            ((cs[:, 1] > len(self.points)-1) & (cs[:, 2] > 0.)))
                    indices = closure_cs.flatten()
                    for index in indices:
                        cs[index, 2:] *= -1
                gc = GaussCode(cs)
                gc.simplify(verbose=False)
                polys.append([angs[0], angs[1], alexander(gc, simplify=False)])

            else:
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

        return sorted(fracs, key=lambda j: j[1])

    def _alexander_map_values(self, number_of_samples=10, interpolation=100,
                              **kwargs):
        polys = self.alexander_polynomials(
            number_of_samples=number_of_samples, **kwargs)

        from scipy.interpolate import griddata

        positions = []
        for i, row in enumerate(polys):
            positions.append(gall_peters(row[0], row[1]))
        positions = n.array(positions)

        interpolation_points = n.mgrid[0:2*n.pi:int(1.57*interpolation)*1j,
                                       -2.:2.:interpolation*1j]
        values = griddata(positions, polys[:, 2],
                          tuple(interpolation_points),
                          method='nearest')

        return positions, values

    def plot_alexander_map(self, number_of_samples=10,
                           scatter_points=False,
                           mode='imshow', interpolation=100,
                           **kwargs):
        '''
        Creates (and returns) a projective diagram showing each
        different Alexander polynomial in a different colour according
        to a closure on a far away point in this direction.

        Parameters
        ----------
        number_of_samples : int
            The number of points on the sphere to close at.
        scatter_points : bool
            If True, plots a dot at each point on the map projection
            where a closure was made.
        mode : str
            'imshow' to plot the pixels of an image, otherwise plots
            filled contours. Defaults to 'imshow'.
        interpolation : int
            The (short) side length of the interpolation grid on which
            the map projection is made. Defaults to 100.
        '''

        positions, values = self._alexander_map_values(
            number_of_samples,
            interpolation=interpolation)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        if mode == 'imshow':
            ax.imshow(values.T, cmap='jet', interpolation='none')
        else:
            ax.contourf(values.T, cmap='jet',
                        levels=[0] + range(3, int(n.max(values)+1.1), 2))
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_xlim(-0.5, 1.565*interpolation)
        ax.set_ylim(-0.5, 0.995*interpolation)

        im_positions = positions*25*float(interpolation / 100.)
        im_positions[:, 0] -= 0.5
        im_positions[:, 1] += 49.*(interpolation / 100.) + 0.5
        if scatter_points:
            ax.scatter(im_positions[:, 0], im_positions[:, 1], color='black',
                       alpha=1, s=1)

        fig.tight_layout()
        fig.show()

        return fig, ax

    def plot_alexander_shell(self, number_of_samples=10, radius=None,
                             zero_centroid=False,
                             sphere_radius_factor=2.,
                             opacity=0.3, **kwargs):
        '''
        Plots the curve in 3d via self.plot(), along with a translucent
        sphere coloured by the type of knot obtained by closing on each
        point.

        Parameters are all passed to :meth:`OpenKnot.alexander_polynomials`,
        except opacity and kwargs which are given to mayavi.mesh, and
        sphere_radius_factor which gives the radius of the enclosing
        sphere in terms of the maximum Cartesian distance of any point
        in the line from the origin.
        '''

        self.plot()

        positions, values = self._alexander_map_values(
            number_of_samples, radius=None, zero_centroid=False)

        thetas = n.arcsin(n.linspace(-1, 1, 100)) + n.pi/2.
        phis = n.linspace(0, 2*n.pi, 157)

        thetas, phis = n.meshgrid(thetas, phis)

        r = sphere_radius_factor*n.max(self.points)
        zs = r*n.cos(thetas)
        xs = r*n.sin(thetas)*n.cos(phis)
        ys = r*n.sin(thetas)*n.sin(phis)

        import mayavi.mlab as may
        may.mesh(xs, ys, zs, scalars=values, opacity=opacity, **kwargs)


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
