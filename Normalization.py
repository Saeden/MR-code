
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
    if center[0] > 0.003 or center[1] > 0.003 or center[2] > 0.003:
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

    return eigenvectors, eigenvalues


def compute_angle(v1, v2):
    """
    Computes the angle between two vectors in radians.
    """
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(cos_angle)

    return angle


def align_eigenvectors_old(mesh):
    eigenvectors, eigenvalues = compute_pca(mesh)
    sort_eig_val = sorted(eigenvalues, reverse=True)
    maj_eig_vec = eigenvectors[np.where(eigenvalues==sort_eig_val[0])][0]

    x = np.asarray([1,0,0])
    y = np.asarray([0,1,0])

    rot_mat_x = get_rot_matrix(x, maj_eig_vec)

    center = get_barycenter(mesh)
    mesh = mesh.rotate(rot_mat_x, center=center)
    #vertices = np.asarray(mesh.vertices)

    #eigenvectors, eigenvalues = compute_pca(mesh)
    #sort_eig_val = sorted(eigenvalues, reverse=True)
    #med_eig_vec = eigenvectors[np.where(eigenvalues == sort_eig_val[1])][0]
    #med_eig_vec = eigenvectors[1]
    #rot_mat_y = get_rot_matrix(y, med_eig_vec)

    #center = get_barycenter(mesh)
    #mesh.rotate(rot_mat_y, center=center)

    return mesh

def align_eigenvectors_broken(mesh):
    mesh_mat = np.asarray(mesh.vertices)
    eig_vec, eig_val = compute_pca(mesh)
    """sort_eig_val = sorted(eig_val)
    maj_eig_vec = eig_vec[np.where(eig_val == sort_eig_val[2])][0]
    med_eig_vec = eig_vec[np.where(eig_val == sort_eig_val[1])][0]
    axis_fr_eig = np.cross(maj_eig_vec, med_eig_vec)
    axis_vects = np.asarray([maj_eig_vec, med_eig_vec, axis_fr_eig])"""
    new_mat = np.dot(mesh_mat, eig_vec)
    new_vec = o3d.utility.Vector3dVector(new_mat)
    mesh.vertices = new_vec
    #faces = np.asarray(mesh.triangles)
    #triangles = np.ascontiguousarray(np.fliplr(faces))
    #mesh.triangles = o3d.utility.Vector3iVector(triangles)
    #mesh.compute_vertex_normals()
    #mesh.compute_triangle_normals()
    return mesh

def align_eigenvectors(mesh):
    mesh_mat = np.asarray(mesh.vertices)
    mesh_mat_x = mesh_mat[:, 0]
    mesh_mat_y = mesh_mat[:, 1]
    mesh_mat_z = mesh_mat[:, 2]

    eig_vec, eig_val = compute_pca(mesh)
    sort_eig_val = sorted(eig_val)
    maj_eig_vec = eig_vec[np.where(eig_val == sort_eig_val[2])][0]
    med_eig_vec = eig_vec[np.where(eig_val == sort_eig_val[1])][0]
    axis_fr_eig = np.cross(maj_eig_vec, med_eig_vec)

    new_mat_x = np.dot()
    #new_mat_y
    #new_mat_z
    axis_vects = np.asarray([maj_eig_vec, med_eig_vec, axis_fr_eig])
    return mesh

def flip_test(mesh):
    f_x = 0
    f_y = 0
    f_z = 0

    for face in np.asarray(mesh.triangles):
        a = np.asarray(mesh.vertices)[face[0]]
        b = np.asarray(mesh.vertices)[face[1]]
        c = np.asarray(mesh.vertices)[face[2]]
        tri_center = np.asarray([(a[0]+b[0]+c[0])/3, (a[1]+b[1]+c[1])/3, (a[2]+b[2]+c[2])/3])

        f_x += np.sign(tri_center[0])*(tri_center[0])**2
        f_y += np.sign(tri_center[1])*(tri_center[1])**2
        f_z += np.sign(tri_center[2])*(tri_center[2])**2

    verts = np.asarray(mesh.vertices)
    for vert in verts:
        vert[0] *= np.sign(f_x)
        vert[1] *= np.sign(f_y)
        vert[2] *= np.sign(f_z)

    mesh.vertices = o3d.utility.Vector3dVector(verts)
    return mesh


def get_rot_matrix(axis, eig_vec):
    axis_rotation = np.cross(eig_vec, axis)
    unit_axis_rot = axis_rotation/np.linalg.norm(axis_rotation)
    angle_diff = compute_angle(eig_vec, axis)
    rotation_vector = angle_diff * unit_axis_rot
    rot_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_vector)
    return rot_matrix



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
    rot_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angle)
    center = get_barycenter(mesh)
    mesh.rotate(rot_matrix, center=center)
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

    #align_eigen_to_axis(mesh, y, eigenvectors[:, 1])
    return mesh


def testing():
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[0, 0, 0])

    mesh, faces = load_mesh_check_type("./benchmark/db/0/m77/m77.off")
    #view_mesh(mesh, draw_coordinates=True)
    mesh_copy = copy.deepcopy(mesh)

    print(f"Mesh barycenter before normalisation:{get_barycenter(mesh)}")
    mesh_norm = normalise_mesh_step2(mesh)
    #view_mesh(mesh_norm, draw_coordinates=True)
    R = mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, np.pi / 4))
    mesh_norm.rotate(R, center=get_barycenter(mesh))
    print(f"Mesh barycenter before rotation:{get_barycenter(mesh_norm)}")

    mesh_copy = copy.deepcopy(mesh_norm)
    #mesh_copy = align_to_eigenvectors(mesh_copy)
    #view_mesh(mesh_copy, draw_coordinates=True)

    align_eigenvectors_broken(mesh_norm)
    mesh_norm.translate(translation=[1.5, 0, 0])
    mesh_align = align_eigenvectors_broken(mesh_copy)
    #mesh_align = align_to_eigenvectors(mesh_copy)
    print(f"\nMesh barycenter after rotation:{get_barycenter(mesh_align)}")
    mesh_flip = flip_test(mesh_align)

    o3d.visualization.draw_geometries([mesh_norm, mesh_flip, coord_frame])


testing()