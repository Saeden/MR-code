"""
To make the application work, install:

trimesh
pyglet
scipy
open3d

"""

from Mesh_Reading import *
from Statistics import *
from utils import get_complete_classification
from Mesh_refining import refine_all_meshes
from Normalization import normalise_step2

def main():
    print("Multimedia Retrieval application")

    filepath = "./benchmark/db/0/m98/m98.off"
    filepath_ply = "./ply_files/colored_airplane.ply"
    filepath_try = "./try.off"
    filepath_ref = "benchmark/db_refined/16/m1682/m1682.off"

    filename = filepath[(filepath.rfind("/") + 1):filepath.rfind(".")]

    show_stats = False
    save_stats_base = False
    refine_meshes = False
    save_stats_refined = False
    normalise_meshes = True
    save_stats_normalised = True


    #--------------------------------------------------------------------

    mesh, face_type = load_mesh_check_type(filepath_ref, faces=False)

    if face_type is not None:
        print("The facetype of this mesh is:", face_type)

    labels_dictionary = get_complete_classification()

    if show_stats:
        show_basic_statistics(mesh, filename, labels_dictionary)

        if save_stats_base:
           save_statistics("./benchmark/db/")

    if refine_meshes:
        refine_all_meshes("./benchmark/db/", target_faces_number=5000)

    if save_stats_refined:
        save_statistics("./benchmark/db_refined/")

    if normalise_meshes:
        normalise_step2("./benchmark/db_refined/")

    if save_stats_normalised:
        save_statistics("./benchmark/db_ref_normalised/")


    #view_mesh(mesh, draw_coordinates=True, show_wireframe=True, aabbox=True)


if __name__ == "__main__":
    main()