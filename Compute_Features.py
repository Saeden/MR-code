import open3d as o3d
from math import pi
from Normalization import compute_pca
from Mesh_Reading import *
from utils import *
from Statistics import show_shape_statistics
import numpy as np


def compute_global_features(mesh):

    area = mesh.get_surface_area()
    volume = mesh.get_volume()

    compactness = (area**2)/(36*pi*(volume**2))
    sphericity = 1/compactness

    # diameter as the minimum of the coordinates of the max_bound - min_bound of the shape.
    diameter = min(mesh.get_max_bound() - mesh.get_min_bound())

    aabbox_volume = o3d.geometry.AxisAlignedBoundingBox.get_axis_aligned_bounding_box(mesh).volume()

    eigenvectors, eigenvalues = compute_pca(mesh)
    eccentricity = abs(max(eigenvalues))/abs(min(eigenvalues))

    print(area, volume, compactness, sphericity, diameter, aabbox_volume, eccentricity)





def test():

    path = "./benchmark/db_ref_normalised/0/m99/m99.off"
    filename = path[(path.rfind("/") + 1):]

    mesh = load_mesh(path)

    """
    This is an attempt to transform the mesh in watertight, otherwise the volume
    could not be computed with open3d. Sadly, it doesn't work.
    In the db_ref_normalised database we have 1733 non watertight shapes and only 61 watertight ones.
    In the original database (db) we had 1717 non watertight shapes and only 77 watertight ones.
    We have to find a way to compute the volume of non-watertight shapes.
    """

    # remove self intersecting triangles
    intersecting = np.asarray(mesh.get_self_intersecting_triangles())
    unique_triangles = np.unique(intersecting)
    print("Number of removed self-intersecting triangles: ", len(unique_triangles))
    mesh.remove_triangles_by_index(unique_triangles)
    mesh.remove_unreferenced_vertices()

    tri_before = len(np.asarray(mesh.triangles))
    # remove non manifold edges
    mesh.remove_non_manifold_edges()
    tri_after = len(np.asarray(mesh.triangles))
    print("Number of non-manifold triangles removed: ", tri_before-tri_after)
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()

    # remove non manifold vertices
    verts = np.asarray(mesh.get_non_manifold_vertices())
    print("Number of non-manifold vertices removed: ", len(verts))
    mesh.remove_vertices_by_index(verts)

    view_mesh(mesh, draw_coordinates=True, show_wireframe=True, aabbox=True)
    show_shape_statistics(mesh, filename)

    compute_global_features(mesh)


#test()






