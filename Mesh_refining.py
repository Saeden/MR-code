import os
from Mesh_Reading import load_mesh_check_type
import open3d as o3d
import shutil


def calculate_number_iteration(actual_faces_number, target_faces_number):

    possible_values = [1, 2, 3, 4]
    error = []

    # calculate the square-error respect to the actual faces after the remesh and the target
    # for each possible iteration value and then select the iteration that
    # produced the lowest error

    for i in possible_values:
        error.append(((target_faces_number - (actual_faces_number*(4**i)))**2))

    index_min = error.index(min(error))

    return index_min + 1


def refine_single_mesh(mesh, target_faces_number=4000):

    actual_faces_number = len(mesh.triangles)
    splitpoint = int(target_faces_number * 0.4)  # eg: 4000 - x = 4x - 4000
    # splitpoint identifies the value for which is the same refine or not refine the mesh, given the target.
    # In fact, not all meshes with < 4000 faces should be refined, since for every value above 1600,
    # if the refining will be done even only once, the number of faces obtained would be
    # far from 4000 compared to the case that we haven't refined the original mesh.
    # 1600 is the split point: 4000 - 1600 = 2400 and 1600 * 4 - 4000 = 2400.
    # The distance from 4000 of the refined vs not refined is the same.

    if actual_faces_number < splitpoint:

        num_iter = calculate_number_iteration(actual_faces_number, target_faces_number)
        new_mesh = mesh.subdivide_midpoint(number_of_iterations=num_iter)

    # for the submeshing, the algorithm would take care about the target number
    # of faces and we don't have to calculate a splitpoint.
    elif actual_faces_number > target_faces_number:

        new_mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces_number)

    else:

        new_mesh = mesh

    return new_mesh


def refine_all_meshes(db_path, target_faces_number=4000):

    path = "./benchmark/db_refined"

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

                print("Refining mesh: ", filename)

                mesh, face_type = load_mesh_check_type(filepath, faces=False)

                new_mesh = refine_single_mesh(mesh, target_faces_number)

                # no need to save the new faces and vertices number since we can
                # run the save_statistics function on this new database
                # to retrieve all the information

                new_root = path + '/' + root[15:]

                os.mkdir(new_root)

                new_filepath = (new_root + "/" + filename)

                o3d.io.write_triangle_mesh(new_filepath, new_mesh)

                print("Refined mesh saved in: ", new_filepath, "\n")

            else:
                continue
