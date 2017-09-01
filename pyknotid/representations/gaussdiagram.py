'''
GaussDiagram
============

.. image:: example_gauss_diagram_k10_93.png
   :alt: A Gauss diagram for the knot 10_93
   :align: center
   :scale: 50%

Class for creating and viewing Gauss diagrams.


API documentation
~~~~~~~~~~~~~~~~~
'''

from __future__ import print_function
import numpy as n


class GaussDiagram(object):
    '''
    Class for containing and manipulating Gauss diagrams.

    Parameters
    ----------
    representation : Another representation of a knot.
    '''

    def __init__(self, representation):
        from pyknotid.representations.gausscode import GaussCode

        gc = representation
        if not isinstance(gc, GaussCode):
            gc = GaussCode(representation)

        self._gauss_code = gc._gauss_code
        if len(self._gauss_code) == 0 or len(self._gauss_code) > 1:
            raise Exception('GaussDiagram can only be created from '
                            'single-component Gausscode')

        self._fig_ax = None
        self._fig_ax = self.plot()

    def plot(self, fig_ax=None):
        '''Plots the Gauss diagram using matplotlib. This is called
        automatically on __init__.

        Returns a tuple of the matplotlib figure and axis.
        '''
        import matplotlib.pyplot as plt
        if fig_ax is None:
            if self._fig_ax is not None:
                fig, ax = self._fig_ax
            else:
                fig, ax = plt.subplots()
        else:
            fig, ax = fig_ax
        ax.clear()

        c1 = plt.Circle((00, 00), radius=100., linewidth=2,
                        color='black',
                        fill=False, antialiased=True)

        gc = self._gauss_code[0]
        num_points = len(gc)

        angles = n.linspace(0, 2*n.pi, num_points+1)[:-1]

        other_angles = {}
        for i, angle in enumerate(angles):
            gc_entry = gc[i, 0]

            circ_pos = 100*n.array([n.cos(angle), n.sin(angle)])
            rad_pos = 1.2*circ_pos
            ax.annotate(str(gc_entry), xy=circ_pos, xytext=rad_pos)

            if gc_entry in other_angles:
                other_angle = other_angles.pop(gc_entry)
                other_pos = 100*n.array([n.cos(other_angle),
                                         n.sin(other_angle)])

                over = gc[i, 1]
                sign = gc[i, 2]
                colour = {-1: 'red', 1: 'blue'}[sign]
                if over > 0:
                    ax.annotate('', xy=other_pos, xytext=circ_pos,
                                arrowprops={'frac': 0.1, 'color': colour})
                else:
                    ax.annotate('', xytext=other_pos, xy=circ_pos,
                                arrowprops={'frac': 0.1, 'color': colour})
            else:
                other_angles[gc_entry] = angle

        ax.add_artist(c1)

        ax.plot(100*n.cos(angles), 100*n.sin(angles), 'o', markersize=8,
                color='black')

        ax.set_xlim(-140, 140)
        ax.set_ylim(-140, 140)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.show()
        return fig, ax
