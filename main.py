"""
To make the application work, install:

trimesh
pyglet
scipy
pyrender
open3d

"""

from Mesh_Reading import *
from Statistics import *

def main():
    print("Multimedia Retrieval application")

    filepath = "./benchmark/db/0//m99/m99.off"
    # filepath = "/Users/danieledigrandi/Desktop/bone.ply"

    mesh = load_mesh(filepath)

    view_mesh(mesh, draw_coordinate_frame=True)

    # show_basic_statistics(mesh)




if __name__ == "__main__":
    main()