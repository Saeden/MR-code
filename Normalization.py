
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


def align_eigenvectors(mesh):
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





def testing():
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[0, 0, 0])


    mesh, faces = load_mesh_check_type("./benchmark/db/0/m99/m99.off")
    #view_mesh(mesh, draw_coordinates=True)
    mesh_copy = copy.deepcopy(mesh)

    print(f"Mesh barycenter before normalisation:{get_barycenter(mesh)}")
    mesh_norm = translate_to_origin(mesh)
    #view_mesh(mesh_norm, draw_coordinates=True)
    R = mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, np.pi / 4))
    mesh_norm.rotate(R, center=get_barycenter(mesh))
    print(f"Mesh barycenter before rotation:{get_barycenter(mesh_norm)}")

    mesh_copy = copy.deepcopy(mesh_norm)
    #mesh_copy = align_to_eigenvectors(mesh_copy)
    #view_mesh(mesh_copy, draw_coordinates=True)

    #align_eigenvectors_broken(mesh_norm)
    mesh_norm.translate(translation=[1.5, 0, 0])
    mesh_align = align_eigenvectors_broken(mesh_copy)
    #mesh_align = align_to_eigenvectors(mesh_copy)
    print(f"\nMesh barycenter after rotation:{get_barycenter(mesh_align)}")
    mesh_flip = flip_test(mesh_align)

    mesh_flip.vertex_normals = o3d.utility.Vector3dVector([])
    mesh_flip.triangle_normals = o3d.utility.Vector3dVector([])
    """new_mesh = tm.Trimesh(vertices=np.asarray(mesh_flip.vertices), faces=np.asarray(mesh.triangles))
    tm.repair.fix_normals(new_mesh)
    tm.repair.fix_inversion(new_mesh)
    mesh_flip.vertices = o3d.utility.Vector3dVector(new_mesh.vertices)
    mesh_flip.triangles = o3d.utility.Vector3iVector(new_mesh.faces)"""


    mesh_flip.compute_vertex_normals()
    mesh_flip.compute_triangle_normals()

    o3d.visualization.draw_geometries([mesh_norm, mesh_flip, coord_frame])


testing()