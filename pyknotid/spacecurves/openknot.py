'''
OpenKnot
========

Class for working with open (linear) curves, that do not form closed
loops. :class:`OpenKnot` provides methods for visualising these curves
and analysing their topology via different kinds of closures.

API documentation
~~~~~~~~~~~~~~~~~

'''

from __future__ import print_function
import numpy as n
import sympy as sym

from pyknotid.spacecurves.spacecurve import SpaceCurve
from pyknotid.spacecurves.knot import Knot
from pyknotid.spacecurves.rotation import get_rotation_angles, rotate_to_top
from collections import Counter

from pyknotid.invariants import alexander
from pyknotid.representations.gausscode import GaussCode
from pyknotid.representations.representation import Representation
from pyknotid.visualise import plot_shell, plot_sphere_shell_vispy


class OpenKnot(SpaceCurve):
    '''
    Class for holding the vertices of a single line that is assumed to
    be an open curve. This class inherits from
    :class:`~pyknotid.spacecurves.spacecurve.SpaceCurve`, replacing any
    default arguments that assume closed curves, and providing methods
    for statistical analysis of knot invariants on projection and closure.

    All knot invariant methods return the results of a sampling over
    many projections of the knot, unless indicated otherwise.
    '''

    def __init__(self, *args, **kwargs):
        super(OpenKnot, self).__init__(*args, **kwargs)
        self._cached_v2 = {}
        self._cached_v3 = {}

    @SpaceCurve.points.setter
    def points(self, points):
        super(OpenKnot, self.__class__).points.fset(self, points)
        self._cached_alexanders = None

    def tangents(self):
        return super(Knot, self).tangents(closed=True)

    def closing_distance(self):
        '''
        Returns the distance between the first and last points.
        '''
        return n.linalg.norm(self.points[-1] - self.points[0])

    def raw_crossings(self, mode='use_max_jump',
                      virtual_closure=False,
                      recalculate=False, try_cython=False):
        '''
        Calls :meth:`pyknotid.spacecurves.spacecurve.SpaceCurve.raw_crossings`,
        but without including the closing line between the last
        and first points (i.e. setting include_closure=False).
        '''
        if not virtual_closure:
            return super(OpenKnot, self).raw_crossings(mode=mode,
                                                       include_closure=False,
                                                       recalculate=recalculate,
                                                       try_cython=try_cython)
        cs = super(OpenKnot, self).raw_crossings(mode=mode,
                                                 include_closure=True,
                                                 recalculate=recalculate,
                                                 try_cython=try_cython)

        if len(cs) > 0:
            closure_cs = n.argwhere(((cs[:, 0] > len(self.points)-1)) |
                                    ((cs[:, 1] > len(self.points)-1)))
            indices = closure_cs.flatten()
            for index in indices:
                cs[index, 2] = 0
        return cs
        

    def __str__(self):
        if self._crossings is not None:
            return '<OpenKnot with {} points, {} crossings>'.format(
                len(self.points), len(self._crossings))
        return '<OpenKnot with {} points>'.format(len(self.points))

    def closures(self, quantity, num_closures=10):
        angles = get_rotation_angles(num_closures)

        if not hasattr(Representation, quantity):
            raise ValueError('Representation has no invariant {}'.format(quantity))

        results = []
        print_dist = int(max(1, 3000. / len(self.points)))
        for i, angs in enumerate(angles):
            print('angs are', angs)
            if i % print_dist == 0:
                self._vprint('\ri = {} / {}'.format(i, len(angles)), False)
            k = Knot(self.points, verbose=False)
            k._apply_matrix(rotate_to_top(*angs))
            k.zero_centroid()
            cs = k.raw_crossings()
            if len(cs) > 0:
                closure_cs = n.argwhere(((cs[:, 0] > len(self.points)-1) & (cs[:, 2] < 0.)) |
                                        ((cs[:, 1] > len(self.points)-1) & (cs[:, 2] > 0.)))
                indices = closure_cs.flatten()
                for index in indices:
                    cs[index, 2:] *= -1
            gc = Representation(cs, verbose=False)
            gc.simplify()
            results.append(getattr(gc, quantity)())
        return results

    def arclength(self):
        '''Calls :meth:`pyknotid.spacecurves.spacecurve.SpaceCurve.arclength`,
        automatically *not* including the closure.
        '''
        return super(OpenKnot, self).arclength(include_closure=False)

    def smooth(self, repeats=1, window_len=10, window='hanning'):
        '''Calls :meth:`pyknotid.spacecurves.spacecurve.SpaceCurve.smooth`,
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
        angles = get_rotation_angles(number_of_samples ** 2)

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
            (0, 0, 0). Defaults to True.

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
            radius = 100 * n.max(self.points)
            # Not guaranteed to give 10* the real radius, but good enough

        print_dist = int(max(1, 3000. / len(self.points)))
        try:
            for i, angs in enumerate(angles):
                if i % print_dist == 0:
                    self._vprint('\ri = {} / {}'.format(i, len(angles)), False)
                k = Knot(self.points, verbose=False)
                k._apply_matrix(rotate_to_top(*angs))
                self.k = k 
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
                    gc = GaussCode(cs, verbose=self.verbose)
                    gc.simplify()
                    polys.append([angs[0], angs[1], alexander(gc, simplify=False)])

                else:
                    points = k.points
                    closure_point = points[-1] + points[0] / 2.
                    closure_point[2] = radius
                    k.points = n.vstack([points, closure_point])

                    polys.append([angs[0], angs[1], k.alexander_polynomial()])
        except IndexError:
            self.failed_case = k

        self._cached_alexanders[
            (number_of_samples, cache_radius)] = n.array(polys)

        return n.array(polys)

    def closure_alexander_polynomial(self, theta=0, phi=0):
        '''Returns the Alexander polynomial of the knot, when projected in
        the z plane after rotating the given theta and phi to the
        North pole.

        Parameters
        ----------
        theta : float
            The sphere angle theta
        phi : float
            The sphere angle phi
        '''
        k = Knot(self.points, verbose=False)
        k._apply_matrix(rotate_to_top(theta, phi))

        cs = k.raw_crossings()
        if len(cs) > 0:
            closure_cs = n.argwhere(((cs[:, 0] > len(self.points)-1) & (cs[:, 2] < 0.)) |
                                    ((cs[:, 1] > len(self.points)-1) & (cs[:, 2] > 0.)))
            indices = closure_cs.flatten()
            for index in indices:
                cs[index, 2:] *= -1
        gc = GaussCode(cs, verbose=False)
        gc.simplify()
        return alexander(gc, simplify=False)
        

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
        # interpolation_points = n.mgrid[0:2 * n.pi:157j,
        #                        -2.:2.:100j]
        # '''interpolation_points = n.mgrid[-n.pi:n.pi:157j,
        #                        -n.pi/2:n.pi/2:100j]'''

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
        '''fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111, projection="mollweide")
        ax.grid(True)'''

        if mode == 'imshow':
            cax = ax.imshow(values.T, cmap='jet', interpolation='none')
            fig.colorbar(cax)
        else:
            ax.contourf(values.T, cmap='jet',
                        levels=[0] + range(3, int(n.max(values) + 1.1), 2))
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



    def virtual_check(self):
        '''Takes an open curve and checks (for the default projection) if its
        Gauss code corresponds to a virtual knot or not. Returns a
        Boolean of this information.

        .. warning:: This only checks the distance by which entries in
                     the Gauss code are separated, it is *not*
                     guaranteed to detect virtual knots.

        Returns
        -------
        virtual : bool
            True if the Gauss code corresponds to a virtual knot. False
            otherwise.

        '''
        gauss_code = self.gauss_code()._gauss_code[0][:, 0]
        l = len(gauss_code)
        total_crossings = l / 2
        crossing_counter = 1
        virtual = False

        for crossing_number in self.gauss_code().crossing_numbers:
            occurences = n.where(gauss_code == crossing_number)[0]
            first_occurence = occurences[0]
            second_occurence = occurences[1]
            crossing_difference = second_occurence - first_occurence

            if crossing_difference % 2 == 0:
                return True
        return False


    def virtual_checks(self, number_of_samples=10, 
                                        zero_centroid=False):
        '''
        Returns a list of virtual Booleans for the curve with a given number 
        if projections taken from directions approximately evenly distributed. 
        A value of True corresponds to the projection giving a virtual knot, 
        with False returned otherwise.

        Parameters
        ----------
        number_of_samples : int
            The number of points on the sphere to project from. Defaults to 10.
        zero_centroid : bool
            Whether to first move the average position of vertices to
            (0, 0, 0). Defaults to False.

        Returns
        -------
        : ndarray
            A number_of_samples by 3 array of angles and virtual Booleans
            (True if virtual, False otherwise)
        '''
        if zero_centroid:
            self.zero_centroid()

        angles = get_rotation_angles(number_of_samples)

        polys = []

        print_dist = int(max(1, 3000. / len(self.points)))
        for i, angs in enumerate(angles):
            if i % print_dist == 0:
                self._vprint('\ri = {} / {}'.format(i, len(angles)), False)
            k = OpenKnot(self.points, verbose=False)
            k._apply_matrix(rotate_to_top(*angs))
            if zero_centroid:
                k.zero_centroid()
            isvirtual = k.virtual_check()
            polys.append([angs[0], angs[1], isvirtual])

        return n.array(polys)

    def virtual_fractions(self, number_of_samples=10, **kwargs):
        '''Returns each of the virtual booleans from
        self.virtual.check.projections, with the fraction of each type.
        '''
        polys = self.virtual_checks(
            number_of_samples=number_of_samples, **kwargs)
        alexs = n.round(polys[:, 2]).astype(n.int)

        fracs = []
        length = float(len(alexs))
        for alex in n.unique(alexs):
            fracs.append((alex, n.sum(alexs == alex) / length))

        return sorted(fracs, key=lambda j: j[1])
        
    def _virtual_map_values(self, number_of_samples=10, **kwargs):
        polys = self.virtual_checks(
            number_of_samples=number_of_samples, **kwargs)

        from scipy.interpolate import griddata

        positions = []
        for i, row in enumerate(polys):
            positions.append(gall_peters(row[0], row[1]))
        positions = n.array(positions)

        interpolation_points = n.mgrid[0:2 * n.pi:157j,
                               -2.:2.:100j]
        values = griddata(positions, polys[:, 2],
                          tuple(interpolation_points),
                          method='nearest')

        return positions, values

    def plot_virtual_map(self, number_of_samples=10,
                         scatter_points=False,
                         mode='imshow', **kwargs):
        '''
        Creates (and returns) a projective diagram showing each
        different virtual Boolean in a different colour according
        to a projection in this direction.
        '''

        positions, values = self._virtual_map_values(number_of_samples)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        if mode == 'imshow':
            cax = ax.imshow(values.T, cmap='jet', interpolation='none')
            fig.colorbar(cax)
        else:
            ax.contourf(values.T, cmap='jet',
                        levels=[0] + range(3, int(n.max(values) + 1.1), 2))
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_xlim(-0.5, 156.5)
        ax.set_ylim(-0.5, 99.5)

        im_positions = positions * 25
        im_positions[:, 0] -= 0.5
        im_positions[:, 1] += 49.5
        if scatter_points:
            ax.scatter(im_positions[:, 0], im_positions[:, 1], color='black',
                       alpha=1, s=1)

        fig.tight_layout()
        fig.show()

        return fig, ax

    def plot_virtual_shell(self, number_of_samples=10,
                           zero_centroid=False,
                           sphere_radius_factor=2.,
                           opacity=0.3, **kwargs):
        '''
        Plots the curve in 3d via self.plot(), along with a translucent
        sphere coloured according to whether or not the projection from this
        point corresponds to a virtual knot or not.

        Parameters are all passed to 
        :meth:`OpenKnot.virtual_checks`, except opacity and kwargs 
        which are given to mayavi.mesh, and sphere_radius_factor which gives 
        the radius of the enclosing sphere in terms of the maximum Cartesian
        distance of any point in the line from the origin.
        '''

        self.plot()

        positions, values = self._virtual_map_values(
            number_of_samples, zero_centroid=False)

        thetas = n.arcsin(n.linspace(-1, 1, 100)) + n.pi / 2.
        phis = n.linspace(0, 2 * n.pi, 157)

        thetas, phis = n.meshgrid(thetas, phis)

        r = sphere_radius_factor * n.max(self.points)
        zs = r * n.cos(thetas)
        xs = r * n.sin(thetas) * n.cos(phis)
        ys = r * n.sin(thetas) * n.sin(phis)

        import mayavi.mlab as may

        may.mesh(xs, ys, zs, scalars=values, opacity=opacity, **kwargs)

    def self_linking(self, theta=0, phi=0):
        '''
        Takes an open curve, finds its Gauss code (for the default projection)
        and calculates its self linking number, J(K). See Kauffman 2004 for
        more information.

        Returns
        -------
        : self_link_counter : int
            The self linking number of the open curve
        '''
        k = OpenKnot(self.points, verbose=False)
        k._apply_matrix(rotate_to_top(theta, phi))

        # gausscode = k.gauss_code(virtual_closure=True)
        # gausscode.simplify()
        # gausscode = gausscode.without_virtual()._gauss_code[0]
        gausscode = k.gauss_code()._gauss_code[0]
        # gausscode = k.gauss_code()._gauss_code[0]
        l = len(gausscode)

        self_linking_counter = 0

        cache = {}

        for index, row in enumerate(gausscode):
            number, over_under, orientation = row
            if number in cache:
                if ((index - cache[number]) % 2) == 0:
                    self_linking_counter += orientation
            else:
                cache[number] = index

        return self_linking_counter
                    
    def closure_alexander(self, theta=0, phi=0):
        from pyknotid.spacecurves.knot import Knot
        k = Knot(self.points, verbose=False)
        k._apply_matrix(rotate_to_top(theta, phi))

        cs = k.raw_crossings()
        if len(cs) > 0:
            closure_cs = n.argwhere(((cs[:, 0] > len(self.points)-1) & (cs[:, 2] < 0.)) |
                                    ((cs[:, 1] > len(self.points)-1) & (cs[:, 2] > 0.)))
            indices = closure_cs.flatten()
            for index in indices:
                cs[index, 2:] *= -1
        gausscode = GaussCode(cs, verbose=False)

        l = len(gausscode)

        from pyknotid.invariants import alexander
        return alexander(gausscode)
                

    def self_linkings(self, number_of_samples=10,
                      zero_centroid=False, **kwargs):
        '''
        Returns a list of self linking numbers for the curve with a given
        number of projections taken from directions approximately evenly
        distributed.

        Parameters
        ----------
        number_of_samples : int
            The number of points on the sphere to project from. Defaults to 10.
        zero_centroid : bool
            Whether to first move the average position of vertices to
            (0, 0, 0). Defaults to False.

        Returns
        -------
        : ndarray
            A number_of_samples by 3 array of angles and self linking number
        '''
        if zero_centroid:
            self.zero_centroid()

        angles = get_rotation_angles(number_of_samples)

        polys = []

        print_dist = int(max(1, 3000. / len(self.points)))
        for i, angs in enumerate(angles):
            if i % print_dist == 0:
                self._vprint('\ri = {} / {}'.format(i, len(angles)), False)
            k = OpenKnot(self.points, verbose=False)
            k._apply_matrix(rotate_to_top(*angs))
            if zero_centroid:
                k.zero_centroid()
            self_linking = k.self_linking()
            polys.append([angs[0], angs[1], self_linking])

        return n.array(polys)

    def self_linking_fractions(self, number_of_samples=10, **kwargs):
        '''Returns each of the self linking numbers from
        self.virtual.self_link.projections, with the fraction of each type.
        '''
        self_linkings = self.self_linkings(
            number_of_samples=number_of_samples, **kwargs)
        self_linkings = n.round(self_linkings[:, 2]).astype(n.int)

        fracs = []
        length = float(len(self_linkings))
        for alex in n.unique(self_linkings):
            fracs.append((alex, n.sum(self_linkings == alex) / length))
        # fracs = n.array(fracs)

        return sorted(fracs, key=lambda j: j[1])
        #return fracs[n.argsort(fracs[:, 1])]
        
    def _self_linking_map_values(self, number_of_samples=10, **kwargs):
        polys = self.self_linkings(
            number_of_samples=number_of_samples, **kwargs)

        from scipy.interpolate import griddata

        positions = []
        for i, row in enumerate(polys):
            positions.append(gall_peters(row[0], row[1]))
        positions = n.array(positions)

        interpolation_points = n.mgrid[0:2 * n.pi:157j,
                               -2.:2.:100j]
        values = griddata(positions, polys[:, 2],
                          tuple(interpolation_points),
                          method='nearest')

        return positions, values

    def plot_self_linking_map(self, number_of_samples=10,
                           scatter_points=False,
                           mode='imshow', **kwargs):
        '''
        Creates (and returns) a projective diagram showing each
        different self linking number in a different colour according
        to a projection in this direction.
        '''

        positions, values = self._self_linking_map_values(number_of_samples)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        if mode == 'imshow':
            cax = ax.imshow(values.T, cmap='jet', interpolation='none')
            fig.colorbar(cax)
        else:
            ax.contourf(values.T, cmap='jet',
                        levels=[0] + range(3, int(n.max(values) + 1.1), 2))
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_xlim(-0.5, 156.5)
        ax.set_ylim(-0.5, 99.5)

        im_positions = positions * 25
        im_positions[:, 0] -= 0.5
        im_positions[:, 1] += 49.5
        if scatter_points:
            ax.scatter(im_positions[:, 0], im_positions[:, 1], color='black',
                       alpha=1, s=1)

        fig.tight_layout()
        fig.show()

        return fig, ax

    def plot_self_linking_shell(self, number_of_samples=100, **kwargs):
        '''
        Plots the curve in 3d via self.plot(), along with a translucent
        sphere coloured by the self linking number obtained by projecting from
        this point.

        Parameters are all passed to 
        :meth:`OpenKnot.virtual_checks`, except opacity and kwargs 
        which are given to mayavi.mesh, and sphere_radius_factor which gives 
        the radius of the enclosing sphere in terms of the maximum Cartesian
        distance of any point in the line from the origin.
        '''
        self.plot(**kwargs)
        plot_shell(self._self_linking_map_values, self.points,
                   number_of_samples=number_of_samples,
                   **kwargs)

    def plot_alexander_shell(self, number_of_samples=100, mode='mesh',
                             radius=None,
                             **kwargs):
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

        self.plot(**kwargs)

        if radius is None:
            self_radii = n.sqrt(n.sum(self.points*self.points, axis=1))
            radius = n.max(self_radii) * 2
        print('radius is', radius)
        if mode == 'mesh':
            plot_sphere_shell_vispy(self.closure_alexander_polynomial,
                                    number_of_samples, number_of_samples,
                                    radius=radius,
                                    **kwargs)
        elif mode == 'crude':
            plot_shell(self._alexander_map_values, self.points,
                       number_of_samples=number_of_samples,
                       **kwargs)


    def alexander_polynomials_multiroots(self, number_of_samples=10,
                                         radius=None,
                                         zero_centroid=False):
        '''
        Returns a list of Alexander polynomials for the knot, closing
        on a sphere of the given radius, with the given number of sample
        points approximately evenly distributed on the sphere. The
        Alexander polynomials are found at three different roots (2, 3
        and 4) and a the knot types corresponding to these roots are
        returned also.

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
            (0, 0, 0). Defaults to True.

        Returns
        -------
        : ndarray
            A number_of_samples by 3 array of angles and alexander
            polynomials.
        '''
        from pyknotid.catalogue.identify import from_invariants
        from pyknotid.catalogue.database import Knot as dbknot

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
            radius = 100 * n.max(self.points)
            # Not guaranteed to give 10* the real radius, but good enough

        print_dist = int(max(1, 3000. / len(self.points)))
        for i, angs in enumerate(angles):
            if i % print_dist == 0:
                self._vprint('\ri = {} / {}'.format(i, len(angles)), False)
            k = Knot(self.points, verbose=False)
            k._apply_matrix(rotate_to_top(*angs))
            if zero_centroid:
                k.zero_centroid()
            points = k.points
            closure_point = points[-1] + points[0] / 2.
            closure_point[2] = radius
            k.points = n.vstack([points, closure_point])
            root_at_two = k.alexander_at_root(2)
            root_at_three = k.alexander_at_root(3)
            root_at_four = k.alexander_at_root(4)
            k_gauss_code = k.gauss_code()
            k_gauss_code.simplify()
            max_crossings = len(k_gauss_code)
            if max_crossings > 17:
                max_crossings = 18
            knot_type = from_invariants(determinant=root_at_two, alex_imag_3=root_at_three,
                                        alex_imag_4=root_at_four,
                                        other=[dbknot.min_crossings <= max_crossings])
            knot_type = knot_db_to_string(knot_type)
            polys.append([angs[0], angs[1], root_at_two, root_at_three, root_at_four, max_crossings,
                          knot_type])

            # self._cached_alexanders[
            #      (number_of_samples, cache_radius)] = n.array(polys)

        return polys

    def multiroots_fractions(self, number_of_samples=10, **kwargs):

        '''
        Returns each of the knot types from
        self.alexander_polynomials_multiroots, with the fraction of that type.
        '''

        knot_info = self.alexander_polynomials_multiroots(
            number_of_samples, **kwargs)
        knot_types = []
        for closures in knot_info:
            knot_types.append(closures[6])
        knot_types = [item for sublist in knot_types for item in sublist] #flattens list
        knot_frequency = Counter(knot_types)
        common = knot_frequency.most_common()
        list_common = [list(elem) for elem in common]
        list_common_fractions = [[elem[0],elem[1]/float(number_of_samples)] for elem in list_common]
        return list_common_fractions

    def generalised_alexander(self):
        '''
        Returns the generalised Alexander polynomial for the default projection
        of the open knot
        '''

        gauss_code_crossings = self.gauss_code()._gauss_code[0][:, 0]
        gauss_code_over_under = self.gauss_code()._gauss_code[0][:,1]
        gauss_code_orientations = self.gauss_code()._gauss_code[0][:,2]
        x = sym.var('x')
        y = sym.var('y')
        m_plus = sym.Matrix([[1 - x, -y], [-x * y**-1, 0]])
        m_minus = sym.Matrix([[0, -x**-1 * y], [-y**-1, 1 - x**-1]])
        num_crossings = len(self.gauss_code())
        matrix = sym.zeros(2 * num_crossings, 2 *num_crossings)
        permutation_matrix = sym.zeros(2 * num_crossings, 2 * num_crossings)

        arc_labels = [0]*len(gauss_code_crossings)
        for i in range(len(gauss_code_crossings)):
            arc_labels[i] = [0] * 4

        counter = 0
        for crossing_number in self.gauss_code().crossing_numbers:
            occurrences = n.where(gauss_code_crossings == crossing_number)[0]
            if gauss_code_orientations[occurrences[0]] == 1:
                m = m_plus
            else:
                m = m_minus
            for i in [0,1]:
                for j in [0,1]:
                    matrix[counter*2 + i,counter*2 + j] = m[i,j]
            counter += 1

        for i in range(len(gauss_code_crossings)):
            arc_labels[i][0] = gauss_code_crossings[i]
            arc_labels[i][1] = (gauss_code_orientations[i] *
                                gauss_code_over_under[i]) #-1 = r, +1 = l
            arc_labels[i-1][2] = gauss_code_crossings[i]
            arc_labels[i-1][3] = (-1 * gauss_code_orientations[i] *
                                  gauss_code_over_under[i]) #-1 = r, +1 = l

        counter = 1
        for crossing_number in self.gauss_code().crossing_numbers:
            for i in range(len(gauss_code_crossings)):
                if arc_labels[i][0] == crossing_number:
                    arc_labels[i][0] = counter
                if arc_labels[i][2] == crossing_number:
                    arc_labels[i][2] = counter
            counter += 1

        for i in range(len(arc_labels)):
            if arc_labels[i][1] < 0:
                if arc_labels[i][3] < 0 :
                    #bottom right
                    permutation_matrix[arc_labels[i][2]*2-1, arc_labels[i][0]*2-1] = 1
                else:
                    #bottom left
                    permutation_matrix[arc_labels[i][2]*2-2, arc_labels[i][0]*2-1] = 1
            else:
                if arc_labels[i][3] < 0:
                    #upper right
                    permutation_matrix[arc_labels[i][2]*2-1, arc_labels[i][0]*2-2] = 1
                else:
                    #upper left
                    permutation_matrix[arc_labels[i][2]*2-2, arc_labels[i][0]*2-2] = 1

        writhe = sum(gauss_code_orientations)/2

        return (-1)**writhe * ((matrix - permutation_matrix).det())

    def projection_invariant(self, **kwargs):
        '''
        First checks if the projection of an open curve is virtual or classical. If virtual,
        a virtual knot invariant is calculated. Otherwise a classical invariant is calculated.
        '''
        is_virtual = self.virtual_check()
        if is_virtual:
            return(['v', self.self_linking()])
        else:
            from pyknotid.invariants import alexander
            return(['c'], alexander(self.gauss_code(), variable=-1,
                                    quadrant='lr',
                                    mode='python', simplify=False))

    def slip_vassiliev_degree_2_average(self, samples=10, recalculate=False, **kwargs):
        from pyknotid.spacecurves.rotation import (get_rotation_angles,
                                                  rotate_to_top)
        from pyknotid.spacecurves import Knot
        angles = get_rotation_angles(samples)
    
        v2s = []
        for theta, phi in angles:
            k = OpenKnot(self.points, verbose=False)
            k._apply_matrix(rotate_to_top(theta, phi))
            v2 = k._slip_vassiliev_degree_2_projection()
            v2s.append(v2)

        result = n.average(n.abs(v2s))
        return result

    def _slip_vassiliev_degree_2_projection(self):
        gc = self.gauss_code()
        from pyknotid.writhes import slip_vassiliev_2
        return slip_vassiliev_2(gc)

    def vassiliev_degree_2_average(self, samples=10, recalculate=False, **kwargs):
        '''Returns the average Vassliev degree 2 invariant calculated by
        averaging its combinatorial value over many different
        projection directions.

        Parameters
        ----------
        samples : int
            The number of directions to average over. Defaults to 10.
        recalculate : bool
            Whether to recalculate the writhe even if a cached result
            is available. Defaults to False.
        **kwargs :
            These are passed directly to :meth:`raw_crossings`.
        '''
        if (self._cached_v2 and
            samples in self._cached_v2 and
            not recalculate):
            return self._cached_v2[samples]

        from pyknotid.spacecurves.rotation import (get_rotation_angles,
                                                  rotate_to_top)
        from pyknotid.spacecurves import Knot
        angles = get_rotation_angles(samples)
    
        v2s = []
        for theta, phi in angles:
            k = OpenKnot(self.points, verbose=False)
            k._apply_matrix(rotate_to_top(theta, phi))
            v2 = k._vassiliev_degree_2_projection()
            v2s.append(v2)

        result = n.average(n.abs(v2s))
        self._cached_v2[samples] = result
        return result

    def _vassiliev_degree_2_projection(self):
        from pyknotid.spacecurves import Knot
        k = Knot(self.points, verbose=False)
        return k.vassiliev_degree_2(simplify=False,
                                    include_closure=False)

    def vassiliev_degree_3_average(self, samples=10, recalculate=False,
                                   signed=True, **kwargs):
        if (self._cached_v3 and
            samples in self._cached_v3 and
            not recalculate):
            return self._cached_v3[samples]

        from pyknotid.spacecurves.rotation import get_rotation_angles, rotate_to_top            
        from pyknotid.spacecurves import Knot
        angles = get_rotation_angles(samples)
    
        v3s = []
        for theta, phi in angles:
            k = OpenKnot(self.points, verbose=False)
            k._apply_matrix(rotate_to_top(theta, phi))
            v3 = k._vassiliev_degree_3_projection()
            v3s.append(v3)

        if signed:
            result = n.average(v3s)
        elif signed is None:
            result = (n.average(v3s), n.average(n.abs(v3s)))
        else:
            result = n.average(n.abs(v3s))
        self._cached_v3[(samples, signed)] = result
        return result

    def _vassiliev_degree_3_projection(self):
        from pyknotid.spacecurves import Knot
        k = Knot(self.points, verbose=False)
        return k.vassiliev_degree_3(simplify=False,
                                    include_closure=False)

    def virtual_vassiliev_degree_3(self):
        from pyknotid.invariants import virtual_vassiliev_degree_3
        return virtual_vassiliev_degree_3(self.gauss_code())

    def _determinants_and_self_linkings(self, number_of_samples=10, radius=None,
                                        recalculate=False,
                                        zero_centroid=False):
        if zero_centroid:
            self.zero_centroid()

        angles = get_rotation_angles(number_of_samples)

        polys = []
        self_linkings = []

        cache_radius = radius

        if radius is None:
            radius = 100 * n.max(self.points)
            # Not guaranteed to give 10* the real radius, but good enough

        print_dist = int(max(1, 3000. / len(self.points)))
        for i, angs in enumerate(angles):
            if i % print_dist == 0:
                self._vprint('\ri = {} / {}'.format(i, len(angles)), False)
            k = Knot(self.points, verbose=False)
            k._apply_matrix(rotate_to_top(*angs))
            if zero_centroid:
                k.zero_centroid()
            cs = k.raw_crossings()
            if len(cs) > 0:
                closure_cs = n.argwhere(((cs[:, 0] > len(self.points)-1) & (cs[:, 2] < 0.)) |
                                        ((cs[:, 1] > len(self.points)-1) & (cs[:, 2] > 0.)))
                indices = closure_cs.flatten()
                for index in indices:
                    cs[index, 2:] *= -1
            gc = GaussCode(cs, verbose=self.verbose)
            gc.simplify()
            polys.append([angs[0], angs[1], alexander(gc, simplify=False)])

            # Remove closing crossings to calculate self linking
            xs, ys = n.where(cs[:, :2] > len(k.points) - 1)
            keeps = n.ones(len(cs), dtype=n.bool)
            for x in xs:
                keeps[x] = False
            cs = cs[keeps]
            gausscode = GaussCode(cs)._gauss_code[0]
            l = len(gausscode)
            self_linking_counter = 0
            cache = {}
            for index, row in enumerate(gausscode):
                number, over_under, orientation = row
                if number in cache:
                    if ((index - cache[number]) % 2) == 0:
                        self_linking_counter += orientation
                else:
                    cache[number] = index
            self_linkings.append((angs[0], angs[1], self_linking_counter))


        return n.array(polys), n.array(self_linkings)

    def _determinant_and_self_linking_fractions(self, number_of_samples=10,
                                                **kwargs):
        polys, self_linkings = self._determinants_and_self_linkings(
            number_of_samples, **kwargs)

        alexs = n.round(polys[:, 2]).astype(n.int)

        fracs = []
        length = float(len(alexs))
        for alex in n.unique(alexs):
            fracs.append((alex, n.sum(alexs == alex) / length))

        det_fracs = sorted(fracs, key=lambda j: j[1])


        self_linkings = n.round(self_linkings[:, 2]).astype(n.int)

        fracs = []
        length = float(len(self_linkings))
        for linking in n.unique(self_linkings):
            fracs.append((linking, n.sum(self_linkings == linking) / length))

        self_linking_fracs = sorted(fracs, key=lambda j: j[1])

        return det_fracs, self_linking_fracs


    def _closure_and_projection_invariants(self, number_of_samples=10, radius=None,
                                           recalculate=False,
                                           zero_centroid=False):
        if zero_centroid:
            self.zero_centroid()

        angles = get_rotation_angles(number_of_samples)

        closure_knotted = []
        projection_virtual = []
        projection_knotted = []

        cache_radius = radius

        if radius is None:
            radius = 100 * n.max(self.points)
            # Not guaranteed to give 10* the real radius, but good enough

        print_dist = int(max(1, 3000. / len(self.points)))
        for i, angs in enumerate(angles):
            if i % print_dist == 0:
                self._vprint('\ri = {} / {}'.format(i, len(angles)), False)
            k = Knot(self.points, verbose=False)
            k._apply_matrix(rotate_to_top(*angs))
            if zero_centroid:
                k.zero_centroid()
            cs = k.raw_crossings()
            if len(cs) > 0:
                closure_cs = n.argwhere(((cs[:, 0] > len(self.points)-1) & (cs[:, 2] < 0.)) |
                                        ((cs[:, 1] > len(self.points)-1) & (cs[:, 2] > 0.)))
                indices = closure_cs.flatten()
                for index in indices:
                    cs[index, 2:] *= -1
            gc = GaussCode(cs, verbose=self.verbose)
            gc.simplify()
            is_knotted = (alexander(gc, simplify=False)) > 1.5
            closure_knotted.append([angs[0], angs[1], is_knotted])

            # Remove closing crossings to calculate self linking
            xs, ys = n.where(cs[:, :2] > len(k.points) - 1)
            keeps = n.ones(len(cs), dtype=n.bool)
            for x in xs:
                keeps[x] = False
            cs = cs[keeps]
            rep = Representation(cs)
            is_virtual = rep.self_linking() != 0
            projection_virtual.append((angs[0], angs[1], is_virtual))

            if not is_virtual:
                is_knotted = alexander(gc, simplify=False) > 1.5
                projection_knotted.append([angs[0], angs[1], is_knotted])
            else:
                projection_knotted.append([angs[0], angs[1], False])


        return (n.array(closure_knotted), n.array(projection_virtual),
                n.array(projection_knotted))

    def _closure_and_projection_knotted_fractions(self, number_of_samples=10,
                                                    **kwargs):
        polys, self_linkings = self._determinants_and_self_linkings(
            number_of_samples, **kwargs)
        ck, pv, pk = self._closure_and_projection_invariants(number_of_samples, **kwargs)

        ck_fraction = n.average(ck[:, -1])
        pv_fraction = n.average(pv[:, -1])

        return ck_fraction, pv_fraction, n.average(pv[:, -1].astype(n.bool) |
                                                   pk[:, -1].astype(n.bool))

    def plot_vassiliev_spectrum(self, angles=100, max_crossings=8):
        v2 = self.vassiliev_degree_2_average()
        v3 = self.vassiliev_degree_3_average()

        from pyknotid.catalogue.identify import from_invariants
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        ks = from_invariants(max_crossings=max_crossings)
        for k in ks:
            kv2 = k.vassiliev_2
            kv3 = k.vassiliev_3

            ax.scatter([kv2], [kv3])

        ax.set_xlabel('v2')
        ax.set_ylabel('v3')

        ax.scatter([v2], [v3], color='red')

        fig.show()

        return fig, ax


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
    theta -= n.pi / 2
    return (phi, 2 * n.sin(theta))

def mollweide(phi, lambda_):
    '''

    Converts spherical coordinates to the Mollweide
    projection of the sphere, an area-preserving projection in
    the shape of an ellipse.

    Parameters
    ----------
    phi : float
        The latitude, in radians.
    lambda_ : float
        The longitude, in radians.
    '''

    if phi == n.pi/2 or phi == -n.pi/2:
        theta_m = phi
    else:
        theta_n = phi
        theta_m = 0
        while abs(theta_m - theta_n) > 0.000001:
            theta_n = theta_m
            theta_m = (theta_n - (2*theta_n + n.sin(2*theta_n) - n.pi * n.sin(phi)) /
                       (2 + 2*n.cos(2*theta_n)))
    x = ((2 * n.sqrt(2)) / (n.pi)) * lambda_ * n.cos(theta_m)
    y = n.sqrt(2) * n.sin(theta_m)
    return(x,y)


def knot_db_to_string(database_object):
    '''
    Takes output from from_invariants() and returns knot type as decimal.
    For example: <Knot 3_1> becomes 3.1 and <Knot K13n1496> becomes 13.1496
    '''
    db_strings = []
    for entries in database_object:
        db_string = str(entries)
        if db_string[6] == 'K':
            db_string = db_string[7:-1]
        else:
            db_string = db_string[6:-1]
            db_string = db_string.replace('_', '.')
        db_strings.append(db_string)
    return db_strings


