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

    filepath = "./benchmark/db/10/m1097/m1097.off"
    #filepath = "./try.off"
    filename = filepath[(filepath.rfind("/") + 1):filepath.rfind(".")]
    filepath_ply = "./ply_files/colored_airplane.ply"

    mesh, face_type = load_mesh_check_type(filepath)

    print("The facetype of this mesh is: ", face_type)

    labels_dictionary = get_complete_classification()

    #show_basic_statistics(mesh, filename, labels_dictionary)

    #save_statistics("./benchmark/db/")

    view_mesh(mesh, draw_coordinates=False, show_wireframe=True, aabbox=False)


    print("1", len(mesh.triangles))
    mesh_ref = mesh.subdivide_loop(number_of_iterations=3)
    print("2", len(mesh_ref.triangles))
    mesh_ref.remove_non_manifold_edges()
    print("3", len(mesh_ref.triangles))
    directory = "./try.off"
    o3d.io.write_triangle_mesh(directory, mesh_ref)

    view_mesh(mesh_ref, draw_coordinates=False, show_wireframe=True, aabbox=False)





    if False:
        print("1", len(mesh.triangles))
        mesh_smp = mesh.simplify_quadric_decimation(target_number_of_triangles=4000)
        print("2", len(mesh_smp.triangles))
        mesh_smp.remove_non_manifold_edges()
        print("3", len(mesh_smp.triangles))
        directory = "./try.off"
        o3d.io.write_triangle_mesh(directory, mesh_smp)

        view_mesh(mesh_smp, draw_coordinates=False, show_wireframe=True, aabbox=False)

if __name__ == "__main__":
    main()