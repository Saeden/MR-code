"""
To make the application work, install:

trimesh
pyglet
scipy
pyrender

"""

from Mesh_Reading import *
from Statistics import *

def main():
    print("Multimedia Retrieval application")

    filepath = "./benchmark/db/0//m99/m99.off"

    mesh = load_mesh(filepath, pyrender_mode=False)

    show_basic_statistics(mesh)

    view_scene(mesh, pyrender_mode=False, show_3Daabb=False)


if __name__ == "__main__":
    main()