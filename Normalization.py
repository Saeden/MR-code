from Mesh_Reading import load_mesh
from utils import create_new_db, save_shape
import open3d as o3d
import numpy as np
import os


def get_barycenter(mesh):
    """
    This function computes the barycenter as a weighted sum between the areas of the triangles of a mesh,
    times the coordinates of the middle point of the considered triangle. This method gives a more
    precise barycenter estimation rather then computing it as an average of the coordinates of the vertices.
    :param mesh: the mesh of which the barycenter has to be computed
    :return: a numpy array with the x, y, z coordinates of the barycenter.
    """
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

    center = np.array([sample_sum[0]/area, sample_sum[1]/area, sample_sum[2]/area])

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

    :param mesh: the mesh that has to be translated to the origin.
    :return: the translated mesh.
    """

    print("\nTranslating the mesh...")

    barycenter = get_barycenter(mesh)
    mesh.translate(translation=-barycenter)
    print("Mesh translated to origin.")

    return mesh


def scale_aabbox_to_unit(mesh):
    """
    Scales the mesh, such that the maximum elongation of its axis-aligned
    bounding box is of unit size.
    The mesh must be located at the origin for it to work properly.
    :param mesh: the mesh that has to be scaled.
    :return: the scaled mesh.
    """

    print("\nScaling the mesh...")

    center = get_barycenter(mesh)
    if center[0] > 0.003 or center[1] > 0.003 or center[2] > 0.003:
        raise ValueError(
            f'Mesh must be centered around the origin, not {center}'
        )
    factor = 1 / max(mesh.get_max_bound() - mesh.get_min_bound())
    mesh.scale(factor, center)
    print("Mesh scaled.")

    return mesh


def normalise_mesh_step2(mesh):
    """
    This function performs all the 4 normalization tasks on a shape, in the order of:
    translation, alignment, flipping and scaling.
    :param mesh: the mesh that has to be normalized.
    :return: the normalized mesh.
    """

    transl_mesh = translate_to_origin(mesh)
    aligned_mesh = align_eigenvectors(transl_mesh)
    flipped_mesh = flip_test(aligned_mesh)
    scaled_mesh = scale_aabbox_to_unit(flipped_mesh)

    scaled_mesh.remove_duplicated_triangles()
    scaled_mesh.remove_duplicated_vertices()
    scaled_mesh.remove_degenerate_triangles()

    return scaled_mesh


def normalise_step2(db_path, mode="all"):
    """
    This function performs a series of normalisation tasks on a database.
    :param db_path: The path of the input database that has to be normalised.
    :param mode: Which type of normalisation should be made. Values: "all", "translation", "alignment", "flipping", "scaling".
    """

    if mode == "translation":
        path = "./benchmark/db_ref_translated"

    elif mode == "alignment":
        path = "./benchmark/db_ref_aligned"

    elif mode == "flipping":
        path = "./benchmark/db_ref_flipped"

    elif mode == "scaling":
        path = "./benchmark/db_ref_scaled"

    # mode 'all':
    else:
        path = "./benchmark/db_ref_normalised"

    create_new_db(path)

    # start of the normalisation:
    for (root, dirs, files) in os.walk(db_path):

        for filename in files:

            if filename.endswith(".off") or filename.endswith(".ply"):
                filepath = root+'/'+filename

                mesh = load_mesh(filepath)

                if mode == "translation":
                    new_mesh = translate_to_origin(mesh)

                elif mode == "alignment":
                    new_mesh = align_eigenvectors(mesh)

                elif mode == "flipping":
                    new_mesh = flip_test(mesh)

                elif mode == "scaling":
                    new_mesh = scale_aabbox_to_unit(mesh)

                # mode 'all':
                else:
                    new_mesh = normalise_mesh_step2(mesh)

                # no need to save the new faces and vertices number since we can
                # run the save_statistics function on this new database
                # to retrieve all the information

                save_shape(filename, path, new_mesh)

            else:
                continue


def compute_pca(mesh):
    """
    This function computes the eigenvectors and eigenvalues of a mesh.
    :param mesh: mesh that has to be used.
    :return: the eigenvectors and eigenvalues of the mesh in input.
    """

    mesh_matrix = np.asarray(mesh.vertices).transpose()
    cov_matrix = np.cov(mesh_matrix)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    return eigenvectors, eigenvalues


def align_eigenvectors(mesh):
    """
    This function alignes the mesh according to the their eigenvectors.
    :param mesh: mesh that has to be aligned.
    :return: the aligned mesh.
    """

    print("\nAligning the mesh...")

    mesh_mat = np.asarray(mesh.vertices)
    center = get_barycenter(mesh)

    eig_vec, eig_val = compute_pca(mesh)
    sort_eig_val = sorted(eig_val)
    index_maj = np.where(eig_val == sort_eig_val[2])[0][0]
    maj_eig_vec = eig_vec[:, index_maj]

    index_med = np.where(eig_val == sort_eig_val[1])[0][0]
    med_eig_vec = eig_vec[:, index_med]
    axis_fr_eig = np.cross(maj_eig_vec, med_eig_vec)

    for point in mesh_mat:

        x = np.dot(point-center, maj_eig_vec)
        y = np.dot(point-center, med_eig_vec)
        z = np.dot(point-center, axis_fr_eig)
        point[0] = x
        point[1] = y
        point[2] = z

    mesh.vertices = o3d.utility.Vector3dVector(mesh_mat)

    print("Mesh aligned.")

    return mesh


def eigenvectors_check(mesh):

    eig_vec, eig_val = compute_pca(mesh)
    sort_eig_val = sorted(eig_val)
    index_maj = np.where(eig_val == sort_eig_val[2])[0][0]
    maj_eig_vec = eig_vec[:, index_maj]

    index_med = np.where(eig_val == sort_eig_val[1])[0][0]
    med_eig_vec = eig_vec[:, index_med]
    maj_cross_med = np.cross(maj_eig_vec, med_eig_vec)

    dot_x = np.absolute(np.dot(maj_eig_vec, [1, 0, 0]))
    dot_y = np.absolute(np.dot(med_eig_vec, [0, 1, 0]))
    dot_z = np.absolute(np.dot(maj_cross_med, [0, 0, 1]))

    value = (dot_x + dot_y + dot_z) / 3

    return value


def flip_test(mesh):
    """
    This function flip the mesh according to the convention discussed in class.
    :param mesh: mesh that has to be flipped.
    :return: the flipped mesh.
    """

    print("\nFlipping the mesh...")

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

    """
    At this point, the mesh could have the normals broken.
    we noticed by performing some experiments that if an odd number of f_x, f_y and f_z is negative,
    (hence, 1 or 3) then the mesh is inside-out, all the normals are pointing inwards. 
    On the other hand, if an even number (hence only 2) of f_x, f_y and f_z is negative, then the normals are correct.
    In the following code, we will perform this check, and invert all the triangles of a shape, if 1 or 3 among
    f_x, f_y and f_z are negative
    """
    lst_f = [np.sign(f_x), np.sign(f_y), np.sign(f_z)]
    neg_f = len([i for i in lst_f if i < 0])

    if neg_f % 2 == 1:
        faces = np.asarray(mesh.triangles)
        triangles = np.ascontiguousarray(np.fliplr(faces))
        mesh.triangles = o3d.utility.Vector3iVector(triangles)

    print("Mesh flipped.")

    return mesh


def flip_test_check(mesh):

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

    if f_x > 0 and f_y > 0 and f_z > 0:
        return 1
    else:
        return -1
