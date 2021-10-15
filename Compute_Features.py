import open3d as o3d
from math import pi
from Normalization import compute_pca
from Mesh_Reading import *
from utils import *
from Statistics import show_shape_statistics
import numpy as np
import trimesh as tm


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


def compute_global_features_tm(mesh):

    area = mesh.area
    volume = mesh.volume

    return area, volume



def test_tm():

    path = "./benchmark/db_ref_normalised/18/m1800/m1800.off"
    filename = path[(path.rfind("/") + 1):]
    mesh = load_mesh_trimesh(path)
    print(type(mesh))

    print("Watertight1: ", mesh.is_watertight)

    mesh.fix_normals()
    tm.repair.fix_inversion(mesh)
    tm.repair.fix_winding(mesh)
    tm.repair.broken_faces(mesh)
    mesh.remove_duplicate_faces()
    mesh.fill_holes()

    print("Watertight2: ", mesh.is_watertight)

    print("Volume?", mesh.is_volume)

    o3d_mesh = mesh.as_open3d

    view_mesh(o3d_mesh, draw_coordinates=True, show_wireframe=True, aabbox=True)

    area, volume = compute_global_features_tm(mesh)

    print("Area, volume (trimesh):", area, volume)

    o3d_mesh = mesh.as_open3d

    view_mesh(o3d_mesh, draw_coordinates=True, show_wireframe=True, aabbox=True)

    show_shape_statistics(o3d_mesh, filename)

    compute_global_features(o3d_mesh)


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


    # remove non manifold edges
    tri_before = len(np.asarray(mesh.triangles))
    mesh.remove_non_manifold_edges()
    tri_after = len(np.asarray(mesh.triangles))
    print("Number of non-manifold triangles removed: ", tri_before-tri_after)

    # remove non manifold vertices
    verts = np.asarray(mesh.get_non_manifold_vertices())
    print("Number of non-manifold vertices removed: ", len(verts))
    mesh.remove_vertices_by_index(verts)

    # remove self intersecting triangles
    intersecting = np.asarray(mesh.get_self_intersecting_triangles())
    unique_triangles = np.unique(intersecting)
    print("Number of removed self-intersecting triangles: ", len(unique_triangles))
    mesh.remove_triangles_by_index(unique_triangles)

    # mesh cleaning
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    view_mesh(mesh, draw_coordinates=True, show_wireframe=True, aabbox=True)
    show_shape_statistics(mesh, filename)

    compute_global_features(mesh)


test_tm()






