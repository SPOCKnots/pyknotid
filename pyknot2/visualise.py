'''
Visualise
=========

This module contains functions for plotting knots, supporting
different toolkits and types of plot.
'''

import vispy

import numpy as n
from colorsys import hsv_to_rgb
from pyknot2.utils import ensure_shape_tuple, vprint
import random

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
        canvas.unfreeze()
        canvas.view = canvas.central_widget.add_view()
        vispy_canvas = canvas
    # if not vispy_canvas.central_widget.children:
    #     vispy_canvas.view = vispy_canvas.central_widget.add_view()
        

def clear_vispy_canvas():
    global vispy_canvas
    if vispy_canvas is None:
        return
    vispy_canvas.unfreeze()
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
                    tube_points=8, **kwargs):
    # Add an extra point to fix tube drawing bug
    last_tangent = points[-1] - points[-2]
    points = n.vstack([points, points[-1] + 0.0001 * last_tangent])
    
    ensure_vispy_canvas()
    if clf:
        clear_vispy_canvas()
    canvas = vispy_canvas
    from vispy import app, scene, color

    if colour is None:
        from colorsys import hsv_to_rgb
        colours = n.linspace(0, 1, len(points))
        colours = n.array([hsv_to_rgb(c, 1, 1) for c in colours])
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
    
    from visualcollection import MeshCollection
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
                    fig_ax=None, show=True):
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
    ax.plot(points[:, 0], points[:, 1])
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
                            opacity=0.3,
                            method='latitude',
                            edge_color=None,
                            cmap='hsv'):
    '''func must be a function of sphere angles theta, phi'''
    
    from vispy.scene import Sphere, ArcballCamera

    s = Sphere(rows=rows, cols=cols, method=method,
               edge_color=edge_color,
               radius=radius)
    mesh = s.mesh
    md = mesh._meshdata
    vertices = md.get_vertices()

    values = n.zeros(len(vertices))

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
    vprint()

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

    colours[:, -1] = opacity

    md.set_vertex_colors(colours)

    ensure_vispy_canvas()
    vispy_canvas.view.camera = ArcballCamera(fov=30)
    vispy_canvas.view.add(s)
    vispy_canvas.show()
                            

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
    from pyknot2.visualise import vispy_canvas
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
        for segment in line:
            if len(segment) < 4:
                continue
            segments.append(segment)
            segment_colours.append(colour)
            segment_radii.append(radius)
            # plot_line_vispy(segment,
            #                 clf=False, colour=colour, **kwargs)
    plot_lines_vispy(segments, colours=segment_colours,
                     tube_radius=segment_radii, **kwargs)
    
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
        '/home/asandy/devel/pyknot2/pyknot2/templates'))
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
