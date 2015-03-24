'''
Visualise
=========

This module contains functions for plotting knots, supporting
different toolkits and types of plot.
'''

import numpy as n
from colorsys import hsv_to_rgb
from pyknot2.utils import ensure_shape_tuple, vprint
import random

vispy_canvas = None

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
        canvas.view = canvas.central_widget.add_view()
        vispy_canvas = canvas

def clear_vispy_canvas():
    global vispy_canvas
    if vispy_canvas is None:
        return
    vispy_canvas.central_widget.remove_widget(vispy_canvas.view)
    vispy_canvas.view = vispy_canvas.central_widget.add_view()

def plot_line_vispy(points, clf=True, tube_radius=0.5,
                    colour=None, zero_centroid=True, **kwargs):
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
        colours = color.Color(colour)

    l = scene.visuals.Tube(points, color=colours,
                           shading='smooth',
                           radius=tube_radius,
                           tube_points=8)
    
    canvas.view.add(l)
    canvas.view.set_camera('turntable', mode='perspective',
                           up='z', distance=3.*n.max(n.max(
                               points, axis=0)))
    if zero_centroid:
        l.transform = scene.transforms.AffineTransform()
        l.transform.translate(-1*n.average(points, axis=0))

    canvas.show()
    # import ipdb
    # ipdb.set_trace()
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

def plot_cell_mayavi(lines, boundary=None, clf=True, **kwargs):
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
            plot_line(segment, mode='mayavi',
                      clf=False, color=colour, **kwargs)
    
    if boundary is not None:
        draw_bounding_box_mayavi(boundary)

def plot_cell_vispy(lines, boundary=None, clf=True, **kwargs):
    if clf:
        clear_vispy_canvas()
    
    hues = n.linspace(0, 1, len(lines) + 1)[:-1]
    colours = [hsv_to_rgb(hue, 1, 1) for hue in hues]
    random.shuffle(colours)
    i = 0
    for (line, colour) in zip(lines, colours):
        vprint('Plotting line {} / {}\r'.format(i, len(lines)-1),
               False)
        i += 1
        for segment in line:
            if len(segment) < 4:
                continue
            plot_line_vispy(segment,
                            clf=False, colour=colour, **kwargs)
    
    if boundary is not None:
        draw_bounding_box_vispy(boundary)
                

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

def draw_bounding_box_vispy(shape, colour=(0, 0, 0), tube_radius=1):
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

    for line in ls:
        plot_line_vispy(line, clf=False, colour='black',
                        zero_centroid=False)

    global vispy_canvas
    print 'setting center', shape[1] / 2.
    vispy_canvas.central_widget.children[0].camera.center = (
        -shape[1] / 2., -shape[1] / 2., -shape[1] / 2.)

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
    
