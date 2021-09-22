"""
To make the application work, install:

trimesh
pyglet
scipy
pyrender

"""

from Mesh_Reading import *

def main():
    print("Multimedia Retrieval application")

    filepath = "./benchmark/db/0//m99/m99.off"
    pyrender_mode = False

    mesh = load_mesh(filepath, pyrender_mode)

    view_scene(mesh, pyrender_mode)


if __name__ == "__main__":
    main()