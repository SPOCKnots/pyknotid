import numpy as n


def plot_line(points, mode='mayavi', clf=True, **kwargs):
    '''
    Plots the given line, using the toolkit given by mode.
    Currently, only mayavi is supported. Future support will
    include vispy, matplotlib?

    kwargs are passed to the toolkit specific function, except for:

    Parameters
    ----------
    points : array-like
        The nx3 array to plot.
    mode : str
        The toolkit to draw with. Defaults to 'mayavi'.
    clf : bool
        Whether the existing figure should be cleared
        before drawing the new one.

    TODO: remove mode, replace with auto toolkit selection
    '''
    if mode == 'mayavi':
        plot_line_mayavi(points, clf=clf, **kwargs)
    else:
        raise Exception('invalid toolkit/mode')


def plot_line_mayavi(points, clf=True, **kwargs):
    import mayavi.mlab as may
    if clf:
        may.clf()
    mus = n.linspace(0, 1, len(points))
    may.plot3d(points[:, 0], points[:, 1], points[:, 2], mus,
               colormap='hsv', **kwargs)
    

def plot_projection(points, crossings=None):
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
    
    if crossings is not None and len(crossings):
        crossings = n.array(crossings)
        ax.plot(crossings[:, 0], crossings[:, 1], 'ro', alpha=0.5)
    fig.show()
