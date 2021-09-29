
from Mesh_Reading import load_mesh_check_type, view_mesh
import open3d as o3d
import numpy as np


def translate_to_origin(mesh):
    """
    Translates the mesh, such that its centroid
    will be in position [0, 0, 0].

    How it works: the function translate take as input a 3D vector of the form [x_t, y_t, z_t]
    The computation that the function does, is the following:
    for each point [x_i, y_i, z_i] in the given mesh, it applies the following transformation:

    x_new = x_i + x_t
    y_new = y_i + y_t
    z_new = z_i + z_t

    and substitutes [x_i, y_i, z_i] with [x_new, y_new, z_new]

    Thus, if the vector [x_t, y_t, z_t] given in input is the barycenter (or centroid),
    the shape will automatically be translated to the origin, since [x_new, y_new, z_new] = [0, 0, 0]
    if [x_t, y_t, z_t] is the centroid, because [x_i, y_i, z_i] = [x_t, y_t, z_t].
    """

    # Sadly, get_center() returns the mean of the vertices coordinates.
    # we still have to make a function that returns the REAL barycenter,
    # taking into account the area of the triangles.
    # This is why I didn't use: mesh.translate(translation=[0, 0, 0], relative=False)
    # which with relative=False would have directly translated the mesh to [0, 0, 0]
    # but we loose the control of which center (get_center() or real barycenter)
    # the algorithm would have used to perform the translation.
    # When we will have the algorithm for the barycenter, it will be enough to write:
    # mesh.translate(translation=-barycenter)

    mesh.translate(translation=-mesh.get_center())
    return mesh


def scale_aabbox_to_unit(mesh):
    """
    Scales the mesh,
    such that it fits tightly in a unit-sized cube.
    The mesh must be located at the origin.
    """
    center = mesh.get_center()
    if center[0] > 0.001 or center[1] > 0.001 or center[2] > 0.001:
        raise ValueError(
            f'Mesh must be centered around the origin, not {center}'
        )
    factor = 1 / max(mesh.get_max_bound() - mesh.get_min_bound())
    print(type(factor))
    mesh.scale(factor)
    return mesh


mesh, faces = load_mesh_check_type("./benchmark/db/0/m99/m99.off")

print("Center before translating:", mesh.get_center())

view_mesh(mesh, draw_coordinates=True, aabbox=True)

centered_mesh = translate_to_origin(mesh)

print("Center after translating:", centered_mesh.get_center())

view_mesh(centered_mesh, draw_coordinates=True, aabbox=True)

box_points = o3d.geometry.AxisAlignedBoundingBox.get_axis_aligned_bounding_box(mesh).get_box_points()
axis = mesh.get_axis_aligned_bounding_box()
print(f"Axis-aligned bounding box vertices coordinates: \n{np.asarray(box_points)}")
print("Axis 2", axis)

boxed_mesh = scale_aabbox_to_unit(centered_mesh)

box_points = o3d.geometry.AxisAlignedBoundingBox.get_axis_aligned_bounding_box(boxed_mesh).get_box_points()
axis = boxed_mesh.get_axis_aligned_bounding_box()
print(f"Axis-aligned bounding box vertices coordinates: \n{np.asarray(box_points)}")
print("Axis 2", axis)

view_mesh(boxed_mesh, draw_coordinates=True, aabbox=True)
