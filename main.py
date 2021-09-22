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

    mesh = load_mesh(filepath)

    view_scene(mesh)


if __name__ == "__main__":
    main()