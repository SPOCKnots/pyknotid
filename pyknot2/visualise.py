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
        plot_line_mayavi(points, **kwargs)
    else:
        raise Exception('invalid toolkit/mode')


def plot_line_mayavi(points, clf=True, **kwargs):
    if clf:
        may.clf()
    import mayavi.mlab as may
    mus = n.linspace(0, 1, len(points))
    may.plot3d(points[:, 0], points[:, 1], points[:, 2], mus,
               colormap='hsv', **kwargs)
    
