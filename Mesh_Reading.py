import trimesh
import open3d as o3d
import os


def load_mesh_check_type(filepath, faces=False):
    """
    Reads the mesh file located at the specified filepath,
    and returns the mesh as an open3d object.
    Furthermore, it converts all .ply files to .off files, by exporting any .ply file
    with trimesh as a temp .off file, and then it loads the .off file.
    Then, it cancels the temp .off file.
    :param filepath: the filepath of the .ply or .off file containing the mesh.
    :return: a TriangleMesh open3d object & the face type present in the mesh (triangle, quad or mix)
    """
    facetype = None
    ply = False
    if filepath.endswith('.ply'):
        mesh = trimesh.load_mesh(filepath)
        trimesh.exchange.export.export_mesh(mesh, './temp.off', 'off')
        filepath = './temp.off'
        ply = True

    if faces:
        file = open(filepath, "r")
        text = file.readlines()
        file.close()
        facetype = check_type(text)


    if filepath.endswith('.off'):
        mesh = o3d.io.read_triangle_mesh(filepath)
    else:
        raise ValueError('Input file must be either .off or .ply format')
    if ply:
        os.remove('./temp.off')


    mesh.compute_vertex_normals()  # compute the light of the mesh

    print("Try to render a mesh with normals (exist: " + str(mesh.has_vertex_normals()) + ")")

    return mesh, facetype

def check_type(text_file):
    tri = False
    quad = False
    vert_face_lst = text_file[1].split()
    for line in text_file[int(vert_face_lst[0]):]:
        if line[0] == '3':
            tri = True
        elif line[0] == '4':
            quad = True

    if tri and quad:
        return "mix"
    elif tri:
        return "triangles"
    elif quad:
        return "quads"
    else:
        raise ValueError('This file does not contain quads or triangles')




def view_mesh(mesh, draw_coordinates=False, show_wireframe=False, aabbox=False):
    """
    Function used to view a mesh.
    :param mesh: the mesh open3d object that has to be displayed.
    :param draw_coordinates: True if the 3-axis (x, y, z) has to be displayed, False otherwise.
    :param aabbox: True if the axis-aligned bounding box of the mesh has to be displayed, False otherwise.
    """
    shape = [mesh]

    # The (x, y, z) axis will be rendered as x-red, y-green, and z-blue arrows
    if draw_coordinates:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[0, 0, 0])
        shape.append(mesh_frame)

    if aabbox:
        box = o3d.geometry.AxisAlignedBoundingBox.get_axis_aligned_bounding_box(mesh)
        shape.append(box)


    o3d.visualization.draw_geometries(shape, mesh_show_wireframe=show_wireframe)










