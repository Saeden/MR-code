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

    filepath = "./benchmark/db/0/m99/m99.off"
    filename = filepath[(filepath.rfind("/") + 1):filepath.rfind(".")]
    filepath_ply = "./ply_files/colored_airplane.ply"

    mesh = load_mesh(filepath)

    labels_dictionary = get_complete_classification()

    show_basic_statistics(mesh, filename, labels_dictionary)

    view_mesh(mesh, draw_coordinates=False, show_wireframe=True, aabbox=True)


if __name__ == "__main__":
    main()