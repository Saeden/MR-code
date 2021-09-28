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

def main():
    print("Multimedia Retrieval application")

    filepath = "./benchmark/db/17/m1709/m1709.off"
    filepath_ply = "./ply_files/colored_airplane.ply"
    filepath_try = "./try.off"

    filename = filepath[(filepath.rfind("/") + 1):filepath.rfind(".")]

    show_stats = False
    save_stats_base = False
    save_stats_refined = True
    refine_meshes = True

    #--------------------------------------------------------------------

    mesh, face_type = load_mesh_check_type(filepath, faces=False)

    if face_type is not None:
        print("The facetype of this mesh is:", face_type)

    labels_dictionary = get_complete_classification()

    if show_stats:
        show_basic_statistics(mesh, filename, labels_dictionary)

        if save_stats_base:
           save_statistics("./benchmark/db/")

    if refine_meshes:
        refine_all_meshes("./benchmark/db/", target_faces_number=4000)

    if save_stats_refined:
        save_statistics("./benchmark/db_refined/")

    view_mesh(mesh, draw_coordinates=False, show_wireframe=True, aabbox=False)


if __name__ == "__main__":
    main()