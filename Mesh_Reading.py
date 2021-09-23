import trimesh
import pyrender
import open3d as o3d
import os


def load_mesh(filepath):
    """
    Reads the mesh file located at the specified file path,
    and returns the mesh as open3d object.
    Args:
        file_path (str): The file path of the mesh, to be read.
    Returns:
        TriangleMesh: The mesh as an open3d object.
    """
    off = False
    if filepath.endswith('.off'):
        mesh = trimesh.load_mesh(filepath)
        trimesh.exchange.export.export_mesh(mesh, './temp.ply', 'ply')
        filepath = './temp.ply'
        off = True
    if filepath.endswith('.ply'):
        mesh = o3d.io.read_triangle_mesh(filepath)
    else:
        raise ValueError('Input file must be either .OFF or .PLY format')
    if off:
        os.remove('./temp.ply')
    return mesh


def view_mesh(mesh, draw_coordinate_frame=False):
    """
    Function used to view a mesh.
    It constructs an object scene and open the passed mesh.
    :param mesh: the mesh object that has to be displayed
    :param pyrender_mode: True if open with pyrender, False if open with trimesh
    :param show_3Daabb: True if 3D axis-aligned bounding box has to be showed, False otherwise
    """
    shape = [mesh]

    # The x, y, z axis will be rendered as red, green, and blue arrows
    #  respectively.
    if draw_coordinate_frame:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            origin=[0, 0, 0])
        shape.append(mesh_frame)

    o3d.visualization.draw_geometries(shape, mesh_show_wireframe=True)

    """
    if pyrender_mode:

        scene = pyrender.Scene()
        scene.add(mesh)
        pyrender.Viewer(scene, use_raymond_lighting=True)

    else:
        
        #OLD ATTEMPT
        #Preview mesh in an OpenGL window using trimesh (need pyglet and scipy to be installed):
        
        if show_3Daabb:
            (mesh + mesh.bounding_box).show(smooth=False, line_settings={'point_size': 20, 'line_width': 1}, flags="wireframe")
        else:
            mesh.show(smooth=False, line_settings={'point_size': 20, 'line_width': 1}, flags="wireframe")

    """









