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

def main():
    print("Multimedia Retrieval application")

    filepath = "./benchmark/db/17/m1708/m1708.off"
    filename = filepath[(filepath.rfind("/") + 1):filepath.rfind(".")]
    filepath_ply = "./ply_files/colored_airplane.ply"

    mesh, face_type = load_mesh_check_type(filepath)

    print("The facetype of this mesh is: "+face_type)

    labels_dictionary = get_complete_classification()

    #show_basic_statistics(mesh, filename, labels_dictionary)

    #save_statistics("./benchmark/db/")

    view_mesh(mesh, draw_coordinates=False, show_wireframe=True, aabbox=False)


if __name__ == "__main__":
    main()