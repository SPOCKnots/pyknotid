
from vispy.scene import Mesh

from vispy.visuals import MeshVisual

import numpy as np

from vispy.scene.visuals import create_visual_node

class MeshCollectionVisual(MeshVisual):
    def __init__(self, visuals):

        vertices = []
        indices = []
        vertex_colors = []

        for visual in visuals:
            if not isinstance(visual, MeshVisual):
                raise ValueError('Non-MeshVisual received')

            md = visual._meshdata
            vertices.append(md.get_vertices())
            indices.append(md.get_faces())
            vertex_colors.append(md.get_vertex_colors())

        cum_lens = np.hstack([[0], np.cumsum(map(len, vertices))[:-1]])
        for cur_len, inds in zip(cum_lens, indices):
            inds += cur_len

        vertices = np.vstack(vertices)
        indices = np.vstack(indices)
        vertex_colors = np.vstack(vertex_colors)

        MeshVisual.__init__(self, vertices, indices,
                            vertex_colors=vertex_colors,
                            shading='smooth')

            
MeshCollection = create_visual_node(MeshCollectionVisual)
