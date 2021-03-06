import trimesh as tm
import open3d as o3d
import os


def load_mesh(filepath):
    """
    Reads the mesh file located at the specified filepath,
    and returns the mesh as an open3d object.
    Furthermore, it converts all .ply files to .off files, by exporting any .ply file
    with trimesh as a temp .off file, and then it loads the .off file.
    Then, it cancels the temp .off file.
    :param filepath: the filepath of the .ply or .off file containing the mesh.
    :return: a TriangleMesh open3d object.
    """

    ply = False
    if filepath.endswith('.ply'):
        mesh = tm.load_mesh(filepath)
        tm.exchange.export.export_mesh(mesh, './temp.off', 'off')
        filepath = './temp.off'
        ply = True

    if filepath.endswith('.off'):
        mesh = o3d.io.read_triangle_mesh(filepath)
    else:
        raise ValueError('Input file must be either .off or .ply format')

    if ply:
        os.remove('./temp.off')

    print("Succesfully rendered:" + filepath)

    return mesh


def load_mesh_trimesh(filepath):
    """
    Reads the mesh file located at the specified filepath,
    and returns the mesh as a trimesh object.
    :param filepath: the filepath of the .ply or .off file containing the mesh.
    :return: a TriangleMesh trimesh object.
    """

    if filepath.endswith('.ply') or filepath.endswith('.off'):
        mesh = tm.load_mesh(filepath)

    else:
        raise ValueError('Input file must be either .off or .ply format')

    print("Succesfully rendered:" + filepath)

    return mesh


def check_type(mesh):
    """
    Check if the mesh is composed by triangles and/or quads.
    :param mesh: the mesh object to analyze.
    :return: whether the mesh is composed by triangles, quads or a mix of them.
    """

    tri = False
    quad = False

    for triangle in mesh.triangles:
        if len(triangle) == 3:
            tri = True
        elif len(triangle) == 4:
            quad = True

    if tri and quad:
        return "mix"
    elif tri:
        return "triangles"
    elif quad:
        return "quads"
    else:
        raise ValueError('This file does not contain quads or triangles')


def view_mesh(mesh, draw_coordinates=False, show_wireframe=True, aabbox=False):
    """
    Function used to view a mesh.
    :param mesh: a list of open3d meshes object that has to be displayed.
    :param draw_coordinates: True if the 3-axis (x, y, z) has to be displayed, False otherwise.
    :param show_wireframe: True if a solid wireframe atop of the shape has to be draw.
    :param aabbox: True if the axis-aligned bounding box of the mesh has to be displayed, False otherwise.
    """

    # compute the light of the mesh
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    shape = [mesh]

    # The (x, y, z) axis will be rendered as x-red, y-green, and z-blue arrows
    if draw_coordinates:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[0, 0, 0])
        shape.append(mesh_frame)

    if aabbox:
        box = o3d.geometry.AxisAlignedBoundingBox.get_axis_aligned_bounding_box(mesh)
        shape.append(box)

    o3d.visualization.draw_geometries(shape, mesh_show_wireframe=show_wireframe)
