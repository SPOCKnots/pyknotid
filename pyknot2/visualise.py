'''
Visualise
=========

This module contains functions for plotting knots, supporting
different toolkits and types of plot.
'''

import numpy as n

def plot_line(points, mode='mayavi', clf=True, **kwargs):
    '''
    Plots the given line, using the toolkit given by mode.  Currently,
    only mayavi is supported. Future support will include vispy,
    matplotlib?

    kwargs are passed to the toolkit specific function, except for:

    TODO: add auto toolkit selection

    Parameters
    ----------
    points : array-like
        The nx3 array to plot.
    mode : str
        The toolkit to draw with. Defaults to 'auto', which will
        automatically pick the first available toolkit from
        ['mayavi', 'matplotlib', 'vispy'].
    clf : bool
        Whether the existing figure should be cleared
        before drawing the new one.
    '''
            
    if mode == 'mayavi':
        plot_line_mayavi(points, clf=clf, **kwargs)
    elif mode == 'vispy':
        plot_line_vispy(points, clf=clf, **kwargs)
    elif mode == 'matplotlib':
        plot_line_matplotlib(points, clf=clf, **kwargs)
    else:
        raise Exception('invalid toolkit/mode')


def plot_line_mayavi(points, clf=True, tube_radius=1., colormap='hsv',
                     **kwargs):
    import mayavi.mlab as may
    if clf:
        may.clf()
    mus = n.linspace(0, 1, len(points))
    may.plot3d(points[:, 0], points[:, 1], points[:, 2], mus,
               colormap=colormap, tube_radius=tube_radius, **kwargs)

def plot_line_matplotlib(points, **kwargs):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(points[:, 0], points[:, 1], points[:, 2])
    fig.show()

def plot_line_vispy(points, **kwargs):
    from vispy import app, scene
    canvas = scene.SceneCanvas(keys='interactive')
    canvas.view = canvas.central_widget.add_view()
    l = scene.visuals.Line(points, color=(1, 0, 0, 1), width=4,
                                  mode='gl', antialias=True)
    #l = scene.visuals.LinePlot(points[:, :2])
    canvas.view.add(l)
    canvas.view.set_camera('turntable', mode='perspective',
                           up='z', distance=1.5*n.max(n.max(
                               points, axis=0)))
    l.transform = scene.transforms.AffineTransform()
    l.transform.translate(-1*n.average(points, axis=0))

    canvas.show()
    return canvas
    

def plot_projection(points, crossings=None, mark_start=False,
                    fig_ax=None):
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
    fig.show()

    return fig, ax
