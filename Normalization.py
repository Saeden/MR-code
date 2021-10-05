
from Mesh_Reading import load_mesh_check_type, view_mesh
import open3d as o3d
import numpy as np
import os
import shutil
import copy


def get_barycenter(mesh):
    area = mesh.get_surface_area()
    sample_sum = np.zeros(3)
    for face in np.asarray(mesh.triangles):
        a = np.asarray(mesh.vertices)[face[0]]
        b = np.asarray(mesh.vertices)[face[1]]
        c = np.asarray(mesh.vertices)[face[2]]
        tri_center = np.asarray([(a[0]+b[0]+c[0])/3, (a[1]+b[1]+c[1])/3, (a[2]+b[2]+c[2])/3])
        tri_area = 0.5*(np.linalg.norm(np.cross((b-a), (c-a))))
        sample_sum[0] += tri_center[0] * tri_area
        sample_sum[1] += tri_center[1] * tri_area
        sample_sum[2] += tri_center[2] * tri_area
        #sample_sum += tri_center/tri_area

    center = np.array([sample_sum[0]/area, sample_sum[1]/area, sample_sum[2]/area])
    #center = sample_sum/area
    return center


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
    #center = mesh.get_center()
    #print("center is: ", center)
    barycenter = get_barycenter(mesh)
    #print("\nthe bary center before translation is: ", barycenter)
    mesh.translate(translation=-barycenter)
    #new_mesh = mesh.translate(translation=barycenter, relative=False)
    return mesh


def scale_aabbox_to_unit(mesh):
    """
    Scales the mesh,
    such that it fits tightly in a unit-sized cube.
    The mesh must be located at the origin.
    """
    center = get_barycenter(mesh)
    if center[0] > 0.0015 or center[1] > 0.0015 or center[2] > 0.0015:
        raise ValueError(
            f'Mesh must be centered around the origin, not {center}'
        )
    factor = 1 / max(mesh.get_max_bound() - mesh.get_min_bound())
    mesh.scale(factor, center)
    return mesh

def normalise_mesh_step2(mesh):
    transl_mesh = translate_to_origin(mesh)
    scaled_mesh = scale_aabbox_to_unit(transl_mesh)
    return scaled_mesh


def normalise_step2(db_path):

    path = "./benchmark/db_ref_normalised"

    if os.path.exists(path):
        shutil.rmtree(path)

    # make the directories that will contain the new db:
    os.mkdir(path)

    directories = range(19)

    for i in directories:
        new_path = path + "/" + str(i)
        os.mkdir(new_path)

    # start of the remeshing:
    for (root, dirs, files) in os.walk(db_path):

        for filename in files:

            if filename.endswith(".off") or filename.endswith(".ply"):
                filepath = root+'/'+filename

                print("Normalising mesh: ", filename)

                mesh, face_type = load_mesh_check_type(filepath, faces=False)

                new_mesh = normalise_mesh_step2(mesh)


                # no need to save the new faces and vertices number since we can
                # run the save_statistics function on this new database
                # to retrieve all the information

                new_root = path + '/' + root[23:]

                os.mkdir(new_root)

                new_filepath = (new_root + "/" + filename)

                o3d.io.write_triangle_mesh(new_filepath, new_mesh, write_vertex_normals=False)

                print("Normalised mesh saved in: ", new_filepath, "\n")

            else:
                continue


def compute_pca(mesh):
    mesh_matrix = np.asarray(mesh.vertices).transpose()
    cov_matrix = np.cov(mesh_matrix)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    return eigenvectors


def compute_angle(v1, v2):
    """
    Computes the angle between two vectors in radians.
    """
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(cos_angle)

    return angle


def align_eigen_to_axis(mesh, axs, ev):
    """
    Aligns one eigen vector to a predefined axis.
    It uses a rotation of the axis-angle representation.
    In order to obtain the axis of rotation we compute the cross product
    Then, we normalize it so we get a unit vector
    Then, the product of the angle and this normalized unit vector
    equals the rotation vector that we can use to align the
    eigenvalues with the axes.
    """
    rot_axis = np.cross(ev, axs)
    unit_rot_axis = rot_axis / np.linalg.norm(rot_axis)
    angle = compute_angle(ev, axs)
    axis_angle = angle * unit_rot_axis
    mesh.rotate(axis_angle, type=o3d.geometry.RotationType.AxisAngle)
    return mesh


def align_to_eigenvectors(mesh):
    """
    Aligns the mesh,
    such that its eigenvectors are the same direction as the axes.
    """
    x = np.asarray([1, 0, 0])
    y = np.asarray([0, 1, 0])

    vertices = np.asarray(mesh.vertices)
    eigenvectors = np.linalg.eigh(np.cov(vertices, rowvar=False))[1]

    align_eigen_to_axis(mesh, x, eigenvectors[:, 2])

    vertices = np.asarray(mesh.vertices)
    eigenvectors = np.linalg.eigh(np.cov(vertices, rowvar=False))[1]

    align_eigen_to_axis(mesh, y, eigenvectors[:, 1])
    return mesh


def testing():
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[0, 0, 0])

    mesh, faces = load_mesh_check_type("./benchmark/db/0/m98/m98.off")

    center_old = get_barycenter(mesh)

    print("Center before translating:", center_old)

    transl_mesh = copy.deepcopy(mesh).translate(translation=-center_old)




    print("Center after translating:", get_barycenter(transl_mesh))





    o3d.visualization.draw_geometries([coord_frame, mesh, transl_mesh])


#testing()