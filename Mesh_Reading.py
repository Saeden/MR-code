import trimesh
import pyrender
import open3d as o3d
import os

def casa():
    """

    :return:
    """

def load_mesh(filepath):
    """
    Reads the mesh file located at the specified filepath,
    and returns the mesh as open3d object.
    Furthermore, it converts all .ply files to .off files, by exporting any .ply file
    with trimesh as a temp .off file, and then it loads the .off file.
    Then, it cancels the temp .off file.
    :param filepath: the filepath of the .ply or .off file containing the mesh.
    :return: a TriangleMesh open3d object
    """

    ply = False
    if filepath.endswith('.ply'):
        mesh = trimesh.load_mesh(filepath)
        trimesh.exchange.export.export_mesh(mesh, './temp.off', 'off')
        filepath = './temp.off'
        ply = True
    if filepath.endswith('.off'):
        mesh = o3d.io.read_triangle_mesh(filepath)
    else:
        raise ValueError('Input file must be either .off or .ply format')
    if ply:
        os.remove('./temp.off')

    mesh.compute_vertex_normals()  # compute the light of the mesh

    return mesh


def view_mesh(mesh, draw_coordinate_frame=False, show_wireframe=False):
    """
    Function used to view a mesh.
    :param mesh: the mesh open3d object that has to be displayed.
    :param draw_coordinate_frame: True if the 3-axis (x, y, z) has to be displayed, False otherwise.
    """
    shape = [mesh]

    # The (x, y, z) axis will be rendered as x-red, y-green, and z-blue arrows
    if draw_coordinate_frame:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[0, 0, 0])
        shape.append(mesh_frame)


    o3d.visualization.draw_geometries(shape, mesh_show_wireframe=show_wireframe)










