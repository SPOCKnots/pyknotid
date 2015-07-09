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

    def closing_distance(self):
        '''
        Returns the distance between the first and last points.
        '''
        return n.linalg.norm(self.points[-1] - self.points[0])

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
            radius = 100*n.max(self.points)
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
                gc = GaussCode(cs, verbose=self.verbose)
                gc.simplify()
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

        self.plot(mode='mayavi')

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

    def plot_alexander_shell_vispy(self, number_of_samples=10, radius=None,
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

        self.plot(mode='vispy')

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

    def virtual_check(self):
        '''
        Takes an open curve and checks (for the default projection) if its 
        Gauss code corresponds to a virtual knot or not. Returns a Boolean of 
        this information.

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
        
        while(virtual == False and crossing_counter < total_crossings + 1): 
            occurences = n.where(gauss_code == crossing_counter)[0]
            first_occurence = occurences[0]
            second_occurence = occurences[1]
            crossing_difference = second_occurence - first_occurence        
                  
            if(crossing_difference % 2 == 0):
                virtual = True
                  
            crossing_counter += 1
                
        return virtual   
        
   
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

        interpolation_points = n.mgrid[0:2*n.pi:157j,
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
            ax.imshow(values.T, cmap='jet', interpolation='none')
        else:
            ax.contourf(values.T, cmap='jet',
                        levels=[0] + range(3, int(n.max(values)+1.1), 2))
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_xlim(-0.5, 156.5)
        ax.set_ylim(-0.5, 99.5)

        im_positions = positions*25
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

        thetas = n.arcsin(n.linspace(-1, 1, 100)) + n.pi/2.
        phis = n.linspace(0, 2*n.pi, 157)

        thetas, phis = n.meshgrid(thetas, phis)

        r = sphere_radius_factor*n.max(self.points)
        zs = r*n.cos(thetas)
        xs = r*n.sin(thetas)*n.cos(phis)
        ys = r*n.sin(thetas)*n.sin(phis)

        import mayavi.mlab as may
        may.mesh(xs, ys, zs, scalars=values, opacity=opacity, **kwargs)
        
        
    def self_linking_in_projection(self):
        '''
        Takes an open curve, finds its Gauss code (for the default projection) 
        and calculates its self linking number, J(K). See Kauffman 2004 for 
        more information. 

        Returns
        -------
        : self_link_counter : int
            The self linking number of the open curve
        '''
    
        gauss_code = self.gauss_code()._gauss_code
        l = len(gauss_code[0][:,0])
        total_crossings = l/2
        crossing_counter = 1
        self_link_counter = 0        
        
        for i in range(0, total_crossings):
            occurences = n.where(gauss_code[0][:,0] == crossing_counter)[0]
            firstoccurence = occurences[0]
            secondoccurence = occurences[1]
            crossingdifference = secondoccurence - firstoccurence        
                  
            if(crossingdifference%2 == 0):
                self_link_counter += 2 * gauss_code[0][occurences[0],2]
                
            crossing_counter += 1          
            
        return self_link_counter   
        
   
    def self_linkings(self, number_of_samples=10, 
                      zero_centroid=False):
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
            slink = k.self_linking_in_projection()
            polys.append([angs[0], angs[1], slink])
            
        return n.array(polys)
        
    def self_linking_fractions(self, number_of_samples=10, **kwargs):
        '''Returns each of the self linking numbers from
        self.virtual.slink.projections, with the fraction of each type.
        '''
        polys = self.self_linkings(
            number_of_samples=number_of_samples, **kwargs)
        alexs = n.round(polys[:, 2]).astype(n.int)

        fracs = []
        length = float(len(alexs))
        for alex in n.unique(alexs):
            fracs.append((alex, n.sum(alexs == alex) / length))
        #fracs = n.array(fracs)

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

        interpolation_points = n.mgrid[0:2*n.pi:157j,
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
        different self linkming number in a different colour according
        to a projection in this direction.
        '''

        positions, values = self._self_linking_map_values(number_of_samples)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        if mode == 'imshow':
            ax.imshow(values.T, cmap='jet', interpolation='none')
        else:
            ax.contourf(values.T, cmap='jet',
                        levels=[0] + range(3, int(n.max(values)+1.1), 2))
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_xlim(-0.5, 156.5)
        ax.set_ylim(-0.5, 99.5)

        im_positions = positions*25
        im_positions[:, 0] -= 0.5
        im_positions[:, 1] += 49.5
        if scatter_points:
            ax.scatter(im_positions[:, 0], im_positions[:, 1], color='black',
                       alpha=1, s=1)
        
        fig.tight_layout()
        fig.show()

        return fig, ax

    def plot_self_linking_shell(self, number_of_samples=10,
                             zero_centroid=False,
                             sphere_radius_factor=2.,
                             opacity=0.3, **kwargs):
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

        self.plot()

        positions, values = self._self_linking_map_values(
            number_of_samples, zero_centroid=False)

        thetas = n.arcsin(n.linspace(-1, 1, 100)) + n.pi/2.
        phis = n.linspace(0, 2*n.pi, 157)

        thetas, phis = n.meshgrid(thetas, phis)

        r = sphere_radius_factor*n.max(self.points)
        zs = r*n.cos(thetas)
        xs = r*n.sin(thetas)*n.cos(phis)
        ys = r*n.sin(thetas)*n.sin(phis)

        import mayavi.mlab as may
        may.mesh(xs, ys, zs, scalars=values, opacity=opacity, **kwargs)
        
    def plot_self_linking_shell_vispy(self, number_of_samples=10,
                                      zero_centroid=False,
                                      sphere_radius_factor=2.,
                                      opacity=0.3, **kwargs):
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

        import pyknot2.visualise as pvis
        pvis.clear_vispy_canvas()

        self.plot()

        positions, scalars = self._self_linking_map_values(
            number_of_samples, zero_centroid=False)

        print('scalars are', scalars)

        thetas = n.arcsin(n.linspace(-1, 1, 100)) + n.pi/2.
        phis = n.linspace(0, 2*n.pi, 157)

        thetas, phis = n.meshgrid(thetas, phis)

        r = sphere_radius_factor*n.max(self.points)
        zs = r*n.cos(thetas)
        xs = r*n.sin(thetas)*n.cos(phis)
        ys = r*n.sin(thetas)*n.sin(phis)

        # import mayavi.mlab as may
        # may.mesh(xs, ys, zs, scalars=values, opacity=opacity, **kwargs)
        from vispy import scene
        from colorsys import hsv_to_rgb
        from vispy import color as vc
        scalars -= n.min(scalars)
        scalars /= n.max(scalars)
        colours = vc.get_colormap('hot')[scalars.reshape(scalars.shape[0] * scalars.shape[1])].rgba
        colours[:, -1] = 0.1
        print('colours are', colours)
        print('ready to make shell')
        shell = scene.visuals.GridMesh(xs, ys, zs, colors=n.array(colours).reshape(list(xs.shape) + [4]))
        print('made shell')
        # shell = scene.visuals.GridMesh(xs, ys, zs, colors=n.random.random(list(xs.shape) + [3]))
        # print(shell._xs.shape, shell.__colors.shape)
        pvis.vispy_canvas.view.add(shell)
        pvis.vispy_canvas.camera = 'arcball'

        
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
