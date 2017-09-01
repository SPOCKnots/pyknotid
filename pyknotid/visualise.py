'''Visualise
=========

Functions for plotting knots, supporting different toolkits and types
of plot.

pyknotid primarily supports `Vispy <http://vispy.org/>`__ as the
plotting mechanism. `Mayavi
<http://docs.enthought.com/mayavi/mayavi/>`__ is semi-supported but
may not always work.

Many of these functions can be called in a more convenient way via
methods of the :doc:`space curve classes <spacecurves/index>`
(e.g. :class:`~pyknotid.spacecurves.knot.Knot`).

API documentation
-----------------

'''

from __future__ import division

import vispy
# vispy.use('PyQt5')

import numpy as n
import numpy as np
from colorsys import hsv_to_rgb
from pyknotid.utils import ensure_shape_tuple, vprint
import random
from colorsys import hsv_to_rgb

vispy_canvas = None

def plot_vispy_cube():
    ensure_vispy_canvas()
    clear_vispy_canvas()
    canvas = vispy_canvas

    from vispy import app, scene, color

    c = scene.visuals.Cube((5, 2, 10),
                           color='blue',
                           edge_color='red')
    canvas.view.add(c)
    canvas.view.camera = scene.ArcballCamera(fov=30, distance=20)
    canvas.show()

def plot_line(points, mode='auto', clf=True, **kwargs):
    '''
    Plots the given line, using the toolkit given by mode.

    kwargs are passed to the toolkit specific function, except for:

    Parameters
    ----------
    points : ndarray
        The nx3 array to plot.
    mode : str
        The toolkit to draw with. Defaults to 'auto', which will
        automatically pick the first available toolkit from
        ['mayavi', 'matplotlib', 'vispy'], or raise an exception
        if none can be imported.
    clf : bool
        Whether the existing figure should be cleared
        before drawing the new one.
    '''
    if mode == 'auto':
        try:
            import vispy
            mode = 'vispy'
        except ImportError:
            pass
    if mode == 'auto':
        try:
            import mayavi.mlab as may
            mode = 'mayavi'
        except ImportError:
            pass
    if mode == 'auto':
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            mode = 'matplotlib'
        except (ImportError, ValueError):
            pass
    if mode == 'auto':
        raise ImportError('Couldn\'t import any of mayavi, vispy, '
                          'or matplotlib\'s 3d axes.')
            
    if mode == 'mayavi':
        plot_line_mayavi(points, clf=clf, **kwargs)
    elif mode == 'vispy':
        plot_line_vispy(points, clf=clf, **kwargs)
    elif mode == 'matplotlib':
        plot_line_matplotlib(points, clf=clf, **kwargs)
    else:
        raise Exception('invalid toolkit/mode')

    
def plot_line_mayavi(points, clf=True, tube_radius=1., colormap='hsv',
                     closed=True,
                     zero_centroid=False,
                     mus=None,
                     **kwargs):
    import mayavi.mlab as may
    if clf:
        may.clf()
    if mus is None:
        mus = n.linspace(0, 1, len(points))
    may.plot3d(points[:, 0], points[:, 1], points[:, 2], mus,
               colormap=colormap, tube_radius=tube_radius, **kwargs)

def plot_line_matplotlib(points, **kwargs):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(points[:, 0], points[:, 1], points[:, 2])
    fig.show()

def ensure_vispy_canvas():
    global vispy_canvas
    if vispy_canvas is None:
        from vispy import app, scene
        canvas = scene.SceneCanvas(keys='interactive', bgcolor='white')
        try:
            canvas.unfreeze()
        except AttributeError:  # depends on Vispy version
            pass
        canvas.view = canvas.central_widget.add_view()
        vispy_canvas = canvas
    # if not vispy_canvas.central_widget.children:
    #     vispy_canvas.view = vispy_canvas.central_widget.add_view()
        

def clear_vispy_canvas():
    global vispy_canvas
    if vispy_canvas is None:
        return
    try:
        vispy_canvas.unfreeze()
    except AttributeError:  # depends on Vispy version
        pass
    vispy_canvas.central_widget.remove_widget(vispy_canvas.view)
    vispy_canvas.view = vispy_canvas.central_widget.add_view()

def vispy_rotate(elevation=0, max_angle=360):
    global vispy_canvas
    cam = vispy_canvas.view.camera
    from vispy.scene import TurntableCamera
    from time import sleep
    vispy_canvas.view.camera = TurntableCamera(
        fov=cam.fov, scale_factor=cam.scale_factor,
        azimuth=0, elevation=elevation)
    try:
        for i in range(max_angle):
            vispy_canvas.view.camera.azimuth = i
            vispy_canvas.update()
            vispy_canvas.events.draw()
            vispy_canvas.swap_buffers()
            sleep(1 / 60.)
    except KeyboardInterrupt:
        pass
    vispy_canvas.view.camera = cam


def plot_line_vispy(points, clf=True, tube_radius=1.,
                    colour=None, zero_centroid=True,
                    closed=False, mus=None,
                    cmap=None,
                    tube_points=8, **kwargs):
    # Add an extra point to fix tube drawing bug
    last_tangent = points[-1] - points[-2]
    points = n.vstack([points, points[-1] + 0.0001 * last_tangent])
    
    ensure_vispy_canvas()
    if clf:
        clear_vispy_canvas()
    canvas = vispy_canvas
    from vispy import app, scene, color

    if isinstance(cmap, str):
        from matplotlib.cm import get_cmap
        mpl_cmap = get_cmap(cmap)
        cmap = lambda v: n.array(mpl_cmap(v))
    cmap = cmap or (lambda c: hsv_to_rgb(c, 1, 1))

    if colour is None:
        colours = n.linspace(0, 1, len(points))
        colours = n.array([cmap(c) for c in colours])
    else:
        colours = color.ColorArray(colour)

    if mus is not None:
        colours = n.array([hsv_to_rgb(c, 1, 1) for c in mus])
            

    l = scene.visuals.Tube(points, color=colours,
                           shading='smooth',
                           radius=tube_radius,
                           tube_points=tube_points,
                           closed=closed)
    
    canvas.view.add(l)
    # canvas.view.camera = 'arcball'
    canvas.view.camera = scene.ArcballCamera(fov=30, distance=7.5*n.max(
        n.abs(points)))
    #canvas.view.camera = scene.TurntableCamera(fov=30)
    if zero_centroid:
        l.transform = scene.transforms.MatrixTransform()
        # l.transform = scene.transforms.AffineTransform()
        l.transform.translate(-1*n.average(points, axis=0))

    canvas.show()
    # import ipdb
    # ipdb.set_trace()
    return canvas
    
def plot_lines_vispy(lines, clf=True, tube_radius=1.,
                     colours=None, zero_centroid=True, tube_points=8,
                     closed=False,
                     **kwargs):
    ensure_vispy_canvas()
    if clf:
        clear_vispy_canvas()
    canvas = vispy_canvas
    from vispy import app, scene, color

    if not isinstance(tube_radius, list):
        tube_radius = [tube_radius for _ in range(len(lines))]

    if colours is None:
        colours = ['purple' for line in lines]

    tubes = []
    for colour, points, radius in zip(colours, lines, tube_radius):

        l = scene.visuals.Tube(points, color=colour,
                               shading='smooth',
                               radius=radius,
                               closed=closed,
                               tube_points=tube_points)
        tubes.append(l)
    
    from .visualcollection import MeshCollection
    collection = MeshCollection(tubes)
    canvas.view.add(collection)
    canvas.view.camera = 'arcball'
    canvas.view.camera.fov = 30
    # canvas.view.camera = scene.TurntableCamera(
    #     fov=90, up='z', distance=1.2*n.max(n.max(
    #         points, axis=0)))

    if zero_centroid:
        l.transform = scene.transforms.MatrixTransform()
        # l.transform = scene.transforms.AffineTransform()
        l.transform.translate(-1*n.average(points, axis=0))

    canvas.show()
    return canvas
    

def plot_projection(points, crossings=None, mark_start=False,
                    fig_ax=None, show=True, mark_points=False):
    '''
    Plot the 2d projection of the given points, with optional
    markers for where the crossings are.

    Parameters
    ----------
    points : array-like
        The nxm array of points in the line, with m >= 2.
    crossings : array-like or None
        The nx2 array of crossing positions. If None, crossings
        are not plotted. Defaults to None.
    '''
    import matplotlib.pyplot as plt

    if fig_ax is not None:
        fig, ax = fig_ax
    else:
        fig, ax = plt.subplots()
    ax.plot(points[:, 0], points[:, 1], 'o-' if mark_points else '-')
    ax.set_xticks([])
    ax.set_yticks([])

    xmin, ymin = n.min(points[:, :2], axis=0)
    xmax, ymax = n.max(points[:, :2], axis=0)
    dx = (xmax - xmin) / 10.
    dy = (ymax - ymin) / 10.

    ax.set_xlim(xmin - dx, xmax + dx)
    ax.set_ylim(ymin - dy, ymax + dy)

    if mark_start:
        ax.plot([points[0, 0]], [points[0, 1]], color='blue',
                marker='o')
    
    if crossings is not None and len(crossings):
        crossings = n.array(crossings)
        ax.plot(crossings[:, 0], crossings[:, 1], 'ro', alpha=0.5)
    if show:
        fig.show()

    return fig, ax


def plot_shell(func, points, mode='auto', **kwargs):
    if mode =='auto':
        try:
            import vispy
            mode = 'vispy'
        except ImportError:
            pass
    if mode == 'auto':
        try:
            import mayavi.mlab as may
            mode = 'mayavi'
        except ImportError:
            pass

    if mode == 'mayavi':
        plot_shell_mayavi(func, points, **kwargs)
    elif mode == 'vispy':
        plot_shell_vispy(func, points, **kwargs)
    else:
        raise ValueError('invalid toolkit/mode')

def plot_shell_mayavi(func,
                      points,
                      number_of_samples=10,
                      zero_centroid=False,
                      sphere_radius_factor=2.,
                      opacity=0.3, **kwargs):
    positions, values = func(
        number_of_samples, zero_centroid=zero_centroid)

    thetas = n.arcsin(n.linspace(-1, 1, 100)) + n.pi / 2.
    phis = n.linspace(0, 2 * n.pi, 157)

    thetas, phis = n.meshgrid(thetas, phis)

    r = sphere_radius_factor * n.max(points)
    zs = r * n.cos(thetas)
    xs = r * n.sin(thetas) * n.cos(phis)
    ys = r * n.sin(thetas) * n.sin(phis)

    import mayavi.mlab as may

    may.mesh(xs, ys, zs, scalars=values, opacity=opacity, **kwargs)

def plot_sphere_shell_vispy(func, rows=100, cols=100,
                            radius=1.,
                            opacity=1.,
                            translation=(0., 0., 0.),
                            method='latitude',
                            edge_color=None,
                            cmap='hsv',
                            smooth=0,
                            cutoff=0.4,
                            cutoff_max=0.8,
                            transparent_side=True,
                            **kwargs):
    '''func must be a function of sphere angles theta, phi'''
    
    from vispy.scene import Sphere, ArcballCamera

    s = Sphere(rows=rows, cols=cols, method=method,
               edge_color=edge_color,
               radius=radius)
    mesh = s.mesh
    md = mesh._meshdata
    vertices = md.get_vertices()
    md.set_vertices(vertices + n.array(translation))

    values = n.zeros(len(vertices))
    opacities = n.ones(len(vertices))

    print('pre')
    for i, vertex in enumerate(vertices):
        if i % 10 == 0:
            vprint('\ri = {} / {}'.format(i, len(vertices)), newline=False)
        vertex = vertex / n.sqrt(n.sum(vertex*vertex))
        theta = n.arccos(vertex[2])
        phi = n.arctan2(vertex[1], vertex[0])

        if n.isnan(theta):
            theta = 0.0
        values[i] = func(theta, phi)
        distance = vertex[1] - cutoff
        distance_frac = distance / (cutoff_max - cutoff)
        opacity = max(0, 1. - distance_frac)
        opacities[i] = 1. if vertex[1] < cutoff else opacity
    vprint()

    colours = n.zeros((len(values), 4))
    max_val = n.max(values)
    min_val = n.min(values)
    unique_values = n.unique(colours)
    # max_val += (1. + 1./len(unique_values))*(max_val - min_val)
    diff = (max_val - min_val)

    import matplotlib.pyplot as plt
    cm = plt.get_cmap(cmap)
    for i in range(len(colours)):
        colours[i] = cm(((values[i] - min_val) / diff))

    colours[:, -1] = opacity

    faces = md.get_faces()
    for si in range(smooth):
        new_colours = [[n.array(row) for _ in range(3)]
                       for row in colours]
        for i, face in enumerate(faces):
            new_colours[face[0]].append(colours[face[1]])
            new_colours[face[0]].append(colours[face[2]])
            new_colours[face[1]].append(colours[face[0]])
            new_colours[face[1]].append(colours[face[2]])
            new_colours[face[2]].append(colours[face[0]])
            new_colours[face[2]].append(colours[face[1]])

        new_colours = n.array([n.average(cs, axis=0) for cs in new_colours])

        colours = new_colours

    colours = [(c[0], c[1], c[2], o) for c, o in zip(colours, opacities)]

    md.set_vertex_colors(colours)

    ensure_vispy_canvas()
    vispy_canvas.view.camera = ArcballCamera(fov=30)
    vispy_canvas.view.add(s)
    vispy_canvas.show()

    return colours
                            

def plot_shell_vispy(func,
                     points,
                     number_of_samples=10,
                     radius=None,
                     zero_centroid=False,
                     sphere_radius_factor=2.,
                     opacity=0.5,
                     cmap='hsv',
                     **kwargs):
    '''func must be a function returning values at angles and points,
    like OpenKnot._alexander_map_values. 
    '''
    
    positions, values = func(
        number_of_samples, radius=radius,
        zero_centroid=zero_centroid)
    
    thetas = n.arcsin(n.linspace(-1, 1, 100)) + n.pi/2.
    phis = n.linspace(0, 2*n.pi, 157)
    
    thetas, phis = n.meshgrid(thetas, phis)
    
    r = sphere_radius_factor*n.max(points)
    zs = r*n.cos(thetas)
    xs = r*n.sin(thetas)*n.cos(phis)
    ys = r*n.sin(thetas)*n.sin(phis)
    
    colours = n.zeros((values.shape[0], values.shape[1], 4))
    max_val = n.max(values)
    min_val = n.min(values)
    unique_values = n.unique(colours)
    max_val += (1. + 1./len(unique_values))*(max_val - min_val)
    diff = (max_val - min_val)

    import matplotlib.pyplot as plt
    cm = plt.get_cmap(cmap)
    for i in range(colours.shape[0]):
        for j in range(colours.shape[1]):
            colours[i, j] = cm(((values[i, j] - min_val) / diff))

    colours[:, :, -1] = opacity

    from vispy.scene import GridMesh
    from pyknotid.visualise import vispy_canvas
    mesh = GridMesh(xs, ys, zs, colors=colours)
    vispy_canvas.view.add(mesh)


def plot_cell(lines, mode='auto', **kwargs):
    if mode == 'auto':
        try:
            import vispy
            mode = 'vispy'
        except ImportError:
            pass
    if mode == 'auto':
        try:
            import mayavi.mlab as may
            mode = 'mayavi'
        except ImportError:
            pass

    if mode == 'mayavi':
        plot_cell_mayavi(lines, **kwargs)
    elif mode == 'vispy':
        plot_cell_vispy(lines, zero_centroid=False, **kwargs)
    else:
        raise ValueError('invalid toolkit/mode')

def plot_cell_mayavi(lines, boundary=None, clf=True, smooth=True,
                     min_seg_length=5, **kwargs):
    import mayavi.mlab as may
    may.clf()

    hues = n.linspace(0, 1, len(lines) + 1)[:-1]
    colours = [hsv_to_rgb(hue, 1, 1) for hue in hues]
    random.shuffle(colours)
    i = 0
    for (line, colour) in zip(lines, colours):
        vprint('Plotting line {} / {}\r'.format(i, len(lines)-1),
               False)
        i += 1
        for segment in line:
            if len(segment) < min_seg_length:
                continue
            plot_line(segment, mode='mayavi',
                      clf=False, color=colour, **kwargs)
    
    if boundary is not None:
        draw_bounding_box_mayavi(boundary, **kwargs)

def draw_bounding_box_mayavi(shape, colour=(0, 0, 0), tube_radius=1, markz=False):
    if shape is not None:
        if isinstance(shape, (float, int)):
            shape = ensure_shape_tuple(shape)
        if len(shape) == 3:
            shape = (0, shape[0], 0, shape[1], 0, shape[2])
    import mayavi.mlab as may

    xmin, xmax, ymin, ymax, zmin, zmax = shape
    ls = []
    ls.append(n.array([[xmax, ymax, zmin],[xmax, ymax, zmax]]))
    ls.append(n.array([[xmax, ymin, zmin],[xmax, ymin, zmax]]))
    ls.append(n.array([[xmin, ymax, zmin],[xmin, ymax, zmax]]))
    ls.append(n.array([[xmin, ymin, zmin],[xmin, ymin, zmax]]))
    ls.append(n.array([[xmin, ymax, zmax],[xmax, ymax, zmax]]))
    ls.append(n.array([[xmin, ymin, zmax],[xmax, ymin, zmax]]))
    ls.append(n.array([[xmin, ymax, zmin],[xmax, ymax, zmin]]))
    ls.append(n.array([[xmin, ymin, zmin],[xmax, ymin, zmin]]))
    ls.append(n.array([[xmax, ymin, zmax],[xmax, ymax, zmax]]))
    ls.append(n.array([[xmin, ymin, zmax],[xmin, ymax, zmax]]))
    ls.append(n.array([[xmax, ymin, zmin],[xmax, ymax, zmin]]))
    ls.append(n.array([[xmin, ymin, zmin],[xmin, ymax, zmin]]))

    ls = [interpolate(p) for p in ls]

    for line in ls:
        may.plot3d(line[:, 0], line[:, 1], line[:, 2],
                   color=colour, tube_radius=tube_radius)

def plot_cell_vispy(lines, boundary=None, clf=True, colours=None,
                    randomise_colours=True, tube_radius=1., **kwargs):
    if clf:
        clear_vispy_canvas()
    
    if colours is None:
        hues = n.linspace(0, 1, len(lines) + 1)[:-1]
        colours = [hsv_to_rgb(hue, 1, 1) for hue in hues]
        if randomise_colours:
            random.shuffle(colours)
    elif not isinstance(colours, (list, tuple)):
        colours = [colours for _ in lines]
    i = 0
    segments = []
    segment_colours = []
    segment_radii = []
    if not isinstance(tube_radius, list):
        tube_radius = [tube_radius for _ in range(len(lines))]
    for (line, colour, radius) in zip(lines, colours, tube_radius):
        vprint('Plotting line {} / {}\r'.format(i, len(lines)-1),
               False)
        i += 1
        if len(colour) == 4 and colour[-1] == 0:
            continue
        for segment in line:
            if len(segment) < 4:
                continue
            segments.append(segment)
            segment_colours.append(colour)
            segment_radii.append(radius)
            # plot_line_vispy(segment,
            #                 clf=False, colour=colour, **kwargs)
    plot_lines_vispy(segments, colours=segment_colours,
                     tube_radius=segment_radii, clf=clf, **kwargs)
    
    if boundary is not None:
        draw_bounding_box_vispy(boundary, tube_radius=tube_radius[0])

                
def draw_bounding_box_vispy(shape, colour=(0, 0, 0), tube_radius=1.):
    if shape is not None:
        if isinstance(shape, (float, int)):
            shape = ensure_shape_tuple(shape)
        if len(shape) == 3:
            shape = (0, shape[0], 0, shape[1], 0, shape[2])

    xmin, xmax, ymin, ymax, zmin, zmax = shape
    ls = []
    ls.append(n.array([[xmax, ymax, zmin],[xmax, ymax, zmax]]))
    ls.append(n.array([[xmax, ymin, zmin],[xmax, ymin, zmax]]))
    ls.append(n.array([[xmin, ymax, zmin],[xmin, ymax, zmax]]))
    ls.append(n.array([[xmin, ymin, zmin],[xmin, ymin, zmax]]))
    ls.append(n.array([[xmin, ymax, zmax],[xmax, ymax, zmax]]))
    ls.append(n.array([[xmin, ymin, zmax],[xmax, ymin, zmax]]))
    ls.append(n.array([[xmin, ymax, zmin],[xmax, ymax, zmin]]))
    ls.append(n.array([[xmin, ymin, zmin],[xmax, ymin, zmin]]))
    ls.append(n.array([[xmax, ymin, zmax],[xmax, ymax, zmax]]))
    ls.append(n.array([[xmin, ymin, zmax],[xmin, ymax, zmax]]))
    ls.append(n.array([[xmax, ymin, zmin],[xmax, ymax, zmin]]))
    ls.append(n.array([[xmin, ymin, zmin],[xmin, ymax, zmin]]))

    ls = [interpolate(p) for p in ls]

    # plot_lines_vispy(ls, colours=['black' for _ in ls],
    #                  tube_radius=tube_radius, zero_centroid=False)
    plot_lines_vispy(ls, clf=False, colours=[colour for _ in ls],
                     zero_centroid=False, tube_radius=tube_radius)
    # for line in ls:
    #     plot_line_vispy(line, clf=False, colour=colour,
    #                     zero_centroid=False, tube_radius=tube_radius)

    global vispy_canvas
    vispy_canvas.central_widget.children[0].camera.center = (
        shape[1] / 2., shape[1] / 2., shape[1] / 2.)
    vispy_canvas.update()

def interpolate(p, num=10):
    p1, p2 = p
    return n.array([n.linspace(i, j, num) for i, j in zip(p1, p2)]).T

def cell_to_povray(filen, lines, shape):
    from jinja2 import Environment, FileSystemLoader
    env = Environment(loader=FileSystemLoader(
        '/home/asandy/devel/pyknotid/pyknotid/templates'))
    template = env.get_template('cell.pov')

    colours = n.linspace(0, 1, len(lines) + 1)[:-1]
    colours = n.array([hsv_to_rgb(c, 1, 1) for c in colours])

    coloured_segments = []
    for line, colour in zip(lines, colours):
        for segment in line:
            if len(segment) > 3:
                coloured_segments.append((segment, colour))

    with open(filen, 'w') as fileh:
        fileh.write(template.render(lines=coloured_segments))
    

def vispy_save_png(filename):
    img = vispy_canvas.render()
    import vispy.io as io
    io.write_png(filename, img)

def plot_sphere_mollweide_vispy(func, circle_points=50, depth=2,
                                edge_color=None,
                                cmap='hsv',
                                smooth=0,
                                mesh='circles',
                                **kwargs):
    '''func must be a function of sphere angles theta, phi'''
    
    # from vispy.scene import Sphere, ArcballCamera
    from vispy.scene import TurntableCamera, Mesh

    if mesh == 'circles':
        vertices, indices = circles_ellipse_mesh(circle_points, depth)
    else:
        vertices, indices = recursive_ellipse_mesh(circle_points, depth)
    vertices[:, 0] *= 2*n.sqrt(2)
    vertices[:, 1] *= n.sqrt(2)
    mesh = Mesh(vertices, indices)

    md = mesh._meshdata
    vertices = md.get_vertices()

    values = n.zeros(len(vertices))

    print('pre')
    thetas = []
    phis = []
    for i, vertex in enumerate(vertices):
        if i % 10 == 0:
            vprint('\ri = {} / {}'.format(i, len(vertices)), newline=False)

        intermediate = n.arcsin(vertex[1] / n.sqrt(2))
        theta = n.arcsin((2*intermediate + n.sin(2*intermediate)) / n.pi)
        phi = n.pi * vertex[0] / (2*n.sqrt(2) * n.cos(intermediate))

        # theta = n.arccos(vertex[2])
        # phi = n.arctan2(vertex[1], vertex[0])

        if n.isnan(theta):
            theta = 0.0
            print('theta', vertex)
        if n.isnan(phi):
            phi = 0.0
            print('phi', vertex)

        thetas.append(theta)
        phis.append(phi)
        values[i] = func(theta + n.pi/2, phi + n.pi)
    vprint()

    print('thetas', n.min(thetas), n.max(thetas))
    print('phis', n.min(phis), n.max(phis))

    colours = n.zeros((len(values), 4))
    max_val = n.max(values)
    min_val = n.min(values)
    unique_values = n.unique(colours)
    max_val += (1. + 1./len(unique_values))*(max_val - min_val)
    diff = (max_val - min_val)

    import matplotlib.pyplot as plt
    cm = plt.get_cmap(cmap)
    for i in range(len(colours)):
        colours[i] = cm(((values[i] - min_val) / diff))

    colours[:, -1] = 1.

    faces = md.get_faces()
    for si in range(smooth):
        new_colours = [[n.array(row) for _ in range(3)]
                       for row in colours]
        for i, face in enumerate(faces):
            new_colours[face[0]].append(colours[face[1]])
            new_colours[face[0]].append(colours[face[2]])
            new_colours[face[1]].append(colours[face[0]])
            new_colours[face[1]].append(colours[face[2]])
            new_colours[face[2]].append(colours[face[0]])
            new_colours[face[2]].append(colours[face[1]])

        new_colours = n.array([n.average(cs, axis=0) for cs in new_colours])

        colours = new_colours

    md.set_vertex_colors(colours)

    ensure_vispy_canvas()
    clear_vispy_canvas()
    vispy_canvas.view.camera = TurntableCamera(fov=0, elevation=90.)
    vispy_canvas.view.add(mesh)
    vispy_canvas.show()

def circles_ellipse_mesh(radial=5, azimuthal=100):
    angles = n.linspace(0, 2*n.pi, azimuthal + 1)[:-1]
    radii = n.linspace(0, 1., radial)

    offsets = n.zeros(len(angles))
    offsets[::2] += 2*n.pi / azimuthal

    vertices = []
    for i, radius in enumerate(radii):
        cur_angles = angles
        if i % 2 == 0:
            cur_angles += 0.5* (2*n.pi) / azimuthal
        points = n.zeros((len(angles), 3))
        points[:, 0] = n.cos(angles) * radius
        points[:, 1] = n.sin(angles) * radius


        vertices.append(points)

    vertices = n.vstack(vertices)

    indices = []
    num_angles = len(angles)
    for num, radius in enumerate(radii[:-1]):
        base_index = num_angles * num
        next_num = num + 1
        for i in range(len(angles)):
            cur_index = base_index + i
            next_index = base_index + (i + 1) % num_angles
            next_r_index_1 = base_index + ((i - 1) % num_angles) + num_angles
            next_r_index_2 = base_index + i + num_angles
            indices.append((cur_index, next_r_index_1, next_r_index_2))
            indices.append((cur_index, next_r_index_2, next_index))

    return vertices, n.array(indices)
                            
def plot_vertices_indices(vertices, indices):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    ax.plot(vertices[:, 0], vertices[:, 1], 'o')

    for triangle in indices:
        ax.plot([vertices[triangle[0]][0],
                 vertices[triangle[1]][0],
                 vertices[triangle[2]][0]],
                [vertices[triangle[0]][1],
                 vertices[triangle[1]][1],
                 vertices[triangle[2]][1]])

    fig.show()
    return fig, ax
                          


def recursive_ellipse_mesh(init_num_points=10, depth=2):
    triangle = n.array([[0., 0.],
                        [1., 0.],
                        [0.5, 1.5]])

    angles = n.linspace(0, 2*n.pi, init_num_points + 1)[:-1]
    xs = n.sin(angles)
    ys = n.cos(angles)
    centre = n.array([0., 0.])
        
    vertices = []
    indices = []
    vertices_dict = {}

    for i in range(len(angles)):
        triangle = n.array([centre, (xs[i], ys[i]),
                            (xs[(i+1) % len(angles)],
                             ys[(i+1) % len(angles)])])
        indices.extend(get_subtriangles(triangle, vertices,
                                        vertices_dict, 0,
                                        depth))
                                                        

    # indices = get_subtriangles(triangle, vertices, {}, 0, depth)

    return (n.array(vertices), n.array(indices))
                         
    

def get_subtriangles(points, vertices, vertices_dict, depth, max_depth):
    if depth > max_depth:
        indices = []
        for point in points:
            point = tuple(point)
            if tuple(point) not in vertices_dict:
                vertices.append(point)
                vertices_dict[point] = len(vertices) - 1
            indices.append(vertices_dict[point])
                
        return [indices]

    arr_points = n.array(points)

    centre = n.average(arr_points, axis=0)
    half_edges = arr_points + 0.5*(n.roll(arr_points, -1, axis=0) - arr_points)

    new_points = n.array([arr_points[0],
                          half_edges[0],
                          arr_points[1],
                          half_edges[1],
                          arr_points[2],
                          half_edges[2]])

    subtriangles = []
    for i in range(6):
        new_triangle = n.array([centre, new_points[i], new_points[(i+1) % 6]])
        subtriangles.extend(get_subtriangles(new_triangle, vertices,
                                             vertices_dict,
                                             depth+1, max_depth))
    return subtriangles


    
def plot_sphere_lambert_vispy(func, circle_points=50, depth=2,
                              edge_color=None,
                              cmap='hsv',
                              smooth=0,
                              mesh='circles',
                              **kwargs):
    '''func must be a function of sphere angles theta, phi'''
    
    # from vispy.scene import sphere, arcballcamera
    from vispy.scene import TurntableCamera, Mesh

    if mesh == 'circles':
        vertices, indices = circles_ellipse_mesh(circle_points, depth)
    else:
        vertices, indices = recursive_ellipse_mesh(circle_points, depth)
    # vertices[:, 0] *= 2*n.sqrt(2)
    # vertices[:, 1] *= n.sqrt(2)
    vertices[:, 0] *= 2
    vertices[:, 1] *= 2
    mesh = Mesh(vertices, indices)

    md = mesh._meshdata
    vertices = md.get_vertices()

    values = n.zeros(len(vertices))

    print('pre')
    thetas = []
    phis = []
    for i, vertex in enumerate(vertices):
        if i % 10 == 0:
            vprint('\ri = {} / {}'.format(i, len(vertices)), newline=False)

        # print('vertex is', vertex)
        # print('radius is', np.sqrt(vertex[0]**2 + vertex[1]**2))
        # print('angle is', np.arctan2(vertex[1], vertex[0]))

        # intermediate = n.arcsin(vertex[1] / n.sqrt(2))
        # theta = n.arcsin((2*intermediate + n.sin(2*intermediate)) / n.pi)
        # phi = n.pi * vertex[0] / (2*n.sqrt(2) * n.cos(intermediate))

        theta = 2*np.arccos(np.sqrt(vertex[0]**2 + vertex[1]**2)/2.)
        phi = np.arctan2(vertex[1], vertex[0])

        # theta = n.arccos(vertex[2])
        # phi = n.arctan2(vertex[1], vertex[0])

        if n.isnan(theta):
            theta = 0.0
            print('theta', vertex)
        if n.isnan(phi):
            phi = 0.0
            print('phi', vertex)

        thetas.append(theta)
        phis.append(phi)
        values[i] = func(theta + n.pi/2, phi + n.pi)
    vprint()

    print('thetas', n.min(thetas), n.max(thetas))
    print('phis', n.min(phis), n.max(phis))

    colours = n.zeros((len(values), 4))
    max_val = n.max(values)
    min_val = n.min(values)
    unique_values = n.unique(colours)
    max_val += (1. + 1./len(unique_values))*(max_val - min_val)
    diff = (max_val - min_val)

    import matplotlib.pyplot as plt
    cm = plt.get_cmap(cmap)
    for i in range(len(colours)):
        colours[i] = cm(((values[i] - min_val) / diff))

    colours[:, -1] = 1.

    faces = md.get_faces()
    for si in range(smooth):
        new_colours = [[n.array(row) for _ in range(3)]
                       for row in colours]
        for i, face in enumerate(faces):
            new_colours[face[0]].append(colours[face[1]])
            new_colours[face[0]].append(colours[face[2]])
            new_colours[face[1]].append(colours[face[0]])
            new_colours[face[1]].append(colours[face[2]])
            new_colours[face[2]].append(colours[face[0]])
            new_colours[face[2]].append(colours[face[1]])

        new_colours = n.array([n.average(cs, axis=0) for cs in new_colours])

        colours = new_colours

    md.set_vertex_colors(colours)

    ensure_vispy_canvas()
    clear_vispy_canvas()
    vispy_canvas.view.camera = TurntableCamera(fov=0, elevation=90.)
    vispy_canvas.view.add(mesh)
    vispy_canvas.show()


def plot_sphere_lambert_sharp_vispy(func, circle_points=50, depth=2,
                                    output_size=500,
                                    edge_color=None,
                                    cmap='brg',
                                    smooth=0,
                                    mesh='circles',
                                    **kwargs):
    '''func must be a function of sphere angles theta, phi'''
    
    from vispy.scene import TurntableCamera, Mesh

    if mesh == 'circles':
        vertices, indices = circles_ellipse_mesh(circle_points, depth)
    else:
        vertices, indices = recursive_ellipse_mesh(circle_points, depth)
    vertices[:, 0] *= 2
    vertices[:, 1] *= 2
    mesh = Mesh(vertices, indices)

    md = mesh._meshdata
    vertices = md.get_vertices()

    values = n.zeros(len(vertices))

    print('pre')
    thetas = []
    phis = []
    for i, vertex in enumerate(vertices):
        if i % 10 == 0:
            vprint('\ri = {} / {}'.format(i, len(vertices)), newline=False)

        theta = 2*np.arccos(np.sqrt(vertex[0]**2 + vertex[1]**2)/2.)
        phi = np.arctan2(vertex[1], vertex[0])

        if n.isnan(theta):
            theta = 0.0
            print('theta', vertex)
        if n.isnan(phi):
            phi = 0.0
            print('phi', vertex)

        thetas.append(theta)
        phis.append(phi)
        values[i] = func(theta + n.pi/2, phi + n.pi)
    vprint()

    return thetas, phis, values

    import svgwrite as svg
    d = svg.Drawing()

    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(cmap)
    
    max_value = n.max(values)
    min_value = n.min(values)
    def normalise(v):
        return (v - min_value) / (max_value - min_value)
    for tri_i, triangle in enumerate(indices):
        v1 = vertices[triangle[0]]
        v2 = vertices[triangle[1]]
        v3 = vertices[triangle[2]]

        c1 = values[triangle[0]]
        c2 = values[triangle[1]]
        c3 = values[triangle[2]]

        offset_x = 2.001412
        offset_y = 2.0214
        v1_x = (v1[0] + offset_x) / 4. * output_size
        v1_y = (v1[1] + offset_y) / 4. * output_size

        v2_x = (v2[0] + offset_x) / 4. * output_size
        v2_y = (v2[1] + offset_y) / 4. * output_size

        v3_x = (v3[0] + offset_x) / 4. * output_size
        v3_y = (v3[1] + offset_y) / 4. * output_size

        c_x = n.average([v1_x, v2_x, v3_x])
        c_y = n.average([v1_y, v2_y, v3_y])

        v12_x = (v1_x + v2_x) / 2.
        v12_y = (v1_y + v2_y) / 2.
        v13_x = (v1_x + v3_x) / 2.
        v13_y = (v1_y + v3_y) / 2.
        v23_x = (v2_x + v3_x) / 2.
        v23_y = (v2_y + v3_y) / 2.

        cmap_1 = cmap(normalise(c1))
        rgb_cmap1 = [int(c*255) for c in cmap_1[:3]]
        d.add(d.polygon([[v1_x, v1_y],
                         [v12_x, v12_y],
                         [c_x, c_y],
                         [v13_x, v13_y]],
                        fill='rgb({},{},{})'.format(*rgb_cmap1),
                        stroke='rgb({},{},{})'.format(*rgb_cmap1),
                        stroke_width='0.5'))

        cmap_2 = cmap(normalise(c2))
        rgb_cmap2 = [int(c*255) for c in cmap_2[:3]]
        d.add(d.polygon([[v2_x, v2_y],
                         [v12_x, v12_y],
                         [c_x, c_y],
                         [v23_x, v23_y]],
                        fill='rgb({},{},{})'.format(*rgb_cmap2),
                        stroke='rgb({},{},{})'.format(*rgb_cmap2),
                        stroke_width='0.5'))

        cmap_3 = cmap(normalise(c3))
        rgb_cmap3 = [int(c*255) for c in cmap_3[:3]]
        d.add(d.polygon([[v3_x, v3_y],
                         [v13_x, v13_y],
                         [c_x, c_y],
                         [v23_x, v23_y]],
                        fill='rgb({},{},{})'.format(*rgb_cmap3),
                        stroke='rgb({},{},{})'.format(*rgb_cmap3),
                        stroke_width='0.5'))

        # print('cols', c1, cmap_1, c2, cmap_2, c3, cmap_3)

    return d

        
                    
        
        


def get_coloured_subtriangles(points, point_values, vertices, values, vertices_dict, depth, max_depth):
    if depth > max_depth:
        indices = []
        for point, value in zip(points, point_values):
            point = tuple(point)
            if tuple(point) not in vertices_dict:
                vertices.append(point)
                values.append(value)
                vertices_dict[point] = len(vertices) - 1
            indices.append(vertices_dict[point])
                
        return [indices]

    arr_points = n.array(points)

    centre = n.average(arr_points, axis=0)
    half_edges = arr_points + 0.5*(n.roll(arr_points, -1, axis=0) - arr_points)

    new_points = n.array([arr_points[0],
                          half_edges[0],
                          arr_points[1],
                          half_edges[1],
                          arr_points[2],
                          half_edges[2]])

    subtriangles = []
    for i in range(6):
        new_triangle = n.array([centre, new_points[i], new_points[(i+1) % 6]])
        subtriangles.extend(get_subtriangles(new_triangle, vertices,
                                             vertices_dict,
                                             depth+1, max_depth))
    return subtriangles


def point_inside_triangle(point, triangle):
    px, py = point
    p1x, p1y = triangle[0]
    p2x, p2y = triangle[1]
    p3x, p3y = triangle[2]
    alpha = (((p2y - p3y)*(px - p3x) + (p3x - p2x)*(py - p3y)) /
             ((p2y - p3y)*(p1x - p3x) + (p3x - p2x)*(p1y - p3y)))
    if alpha < 0:
        return False
    beta = (((p3y - p1y)*(px - p3x) + (p1x - p3x)*(py - p3y)) /
            ((p2y - p3y)*(p1x - p3x) + (p3x - p2x)*(p1y - p3y)))
    if beta < 0:
        return False
    gamma = 1.0 - alpha - beta
    if gamma < 0:
        return False

    return True



def PointInsideTriangle2(pt,tri):
    '''checks if point pt(2) is inside triangle tri(3x2). @Developer

    source: https://stackoverflow.com/a/20949123/2469283
    '''
    a = 1/(-tri[1,1]*tri[2,0]+tri[0,1]*(-tri[1,0]+tri[2,0])+ \
        tri[0,0]*(tri[1,1]-tri[2,1])+tri[1,0]*tri[2,1])
    s = a*(tri[2,0]*tri[0,1]-tri[0,0]*tri[2,1]+(tri[2,1]-tri[0,1])*pt[0]+ \
        (tri[0,0]-tri[2,0])*pt[1])
    if s<0: return False
    else: t = a*(tri[0,0]*tri[1,1]-tri[1,0]*tri[0,1]+(tri[0,1]-tri[1,1])*pt[0]+ \
              (tri[1,0]-tri[0,0])*pt[1])
    return ((t>0) and (1-s-t>0))
