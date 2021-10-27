"""
To make the application work, install:

trimesh
pyglet
scipy
open3d (not sure if it works for python > 3.7.7)
os
csv
math
numpy
shutil
copy

"""
from Mesh_Reading import load_mesh, view_mesh
from Statistics import *
from utils import get_path, get_read_params, which_database
from Mesh_refining import refine_single_mesh, refine_all_meshes
from Normalization import *


def menu():

    number_of_choices = 4
    possible_choices = [str(i) for i in range(number_of_choices+1)]
    print("\n\n1) Read and visualize a single shape")
    print("2) Re-meshing operations")
    print("3) Normalizing operations")
    print("4) View/save statistics")
    print("\nPress 0 to exit.")
    choice = input("Choice: ")

    while choice not in possible_choices:
        print("Error! Invalid choice")
        choice = input("\nChoice: ")

    return int(choice)


def choice_1():

    path = ""
    # path = "./ply_files/colored_airplane.ply"

    if path == "":
        path = get_path()

    draw_coordinates, aabbox = get_read_params()

    mesh = load_mesh(path)

    view_mesh(mesh, draw_coordinates=draw_coordinates, show_wireframe=True, aabbox=aabbox)


def choice_2():

    number_of_choices = 2
    possible_choices = [str(i) for i in range(number_of_choices+1)]
    print("\n\n1) Re-mesh one shape and visualize the result")
    print("2) Re-mesh all shapes in a database")
    print("\nPress 0 to go back.")
    choice = input("\nChoice: ")

    while choice not in possible_choices:
        print("\nError! Invalid choice")
        choice = input("\nChoice: ")

    choice = int(choice)

    if choice == 1:

        path = get_path()

        draw_coordinates, aabbox = get_read_params()
        mesh = load_mesh(path)
        new_mesh = refine_single_mesh(mesh)

        view_mesh(new_mesh, draw_coordinates=draw_coordinates, show_wireframe=True, aabbox=aabbox)

    elif choice == 2:

        database = which_database()
        path = "./benchmark/" + database

        refine_all_meshes(path)

        print("The new database has been saved at the path ./benchmark/db_refined")


def choice_3():

    number_of_choices = 2
    possible_choices = [str(i) for i in range(number_of_choices+1)]
    print("\n\n1) Perform one or all normalization tasks on one shape")
    print("2) Perform one or all normalization tasks on a database")
    print("\nPress 0 to go back.")
    choice = input("Choice: ")

    while choice not in possible_choices:
        print("\nError! Invalid choice")
        choice = input("\nChoice: ")

    choice = int(choice)

    if choice == 1:

        number_of_choices3_1 = 5
        possible_choices3_1 = [str(i) for i in range(number_of_choices3_1 + 1)]
        print("\nWhich normalization task you want to perform on a shape?")

        print("1) Translation")
        print("2) Alignment")
        print("3) Flipping")
        print("4) Scaling")
        print("5) All")

        choice3_1 = input("Choice: ")

        while choice3_1 not in possible_choices3_1:
            print("\nError! Invalid choice")
            choice3_1 = input("\nChoice: ")

        choice3_1 = int(choice3_1)

        print("On which shape you want to perform that normalization?")

        path = get_path()
        mesh = load_mesh(path)

        if choice3_1 == 1:
            new_mesh = translate_to_origin(mesh)

        elif choice3_1 == 2:
            new_mesh = align_eigenvectors(mesh)

        elif choice3_1 == 3:
            new_mesh = flip_test(mesh)

        elif choice3_1 == 4:
            new_mesh = scale_aabbox_to_unit(mesh)

        elif choice3_1 == 5:
            new_mesh = normalise_mesh_step2(mesh)

        draw_coordinates, aabbox = get_read_params()
        view_mesh(new_mesh, draw_coordinates=draw_coordinates, show_wireframe=True, aabbox=aabbox)


    elif choice == 2:

        number_of_choices3_2 = 5
        possible_choices3_2 = [str(i) for i in range(number_of_choices3_2 + 1)]
        print("\nWhich normalization task you want to perform on a database?")

        print("1) Translation")
        print("2) Alignment")
        print("3) Flipping")
        print("4) Scaling")
        print("5) All")

        choice3_2 = input("Choice: ")

        while choice3_2 not in possible_choices3_2:
            print("\nError! Invalid choice")
            choice3_2 = input("\nChoice: ")

        choice3_2 = int(choice3_2)

        database = which_database()
        path = "./benchmark/" + database

        if choice3_2 == 1:
            normalise_step2(path, mode="translation")

        elif choice3_2 == 2:
            normalise_step2(path, mode="alignment")

        elif choice3_2 == 3:
            normalise_step2(path, mode="flipping")

        elif choice3_2 == 4:
            normalise_step2(path, mode="scaling")

        elif choice3_2 == 5:
            normalise_step2(path, mode="all")


def choice_4():

    number_of_choices = 2
    possible_choices = [str(i) for i in range(number_of_choices + 1)]
    print("\n\n1) Show statistics of one shape")
    print("2) Save statistics of a database")
    print("\nPress 0 to go back.")
    choice = input("\nChoice: ")

    while choice not in possible_choices:
        print("\nError! Invalid choice")
        choice = input("\nChoice: ")

    choice = int(choice)

    if choice == 1:

        path = get_path()
        filename = path[(path.rfind("/") + 1):]

        mesh = load_mesh(path)

        show_shape_statistics(mesh, filename)

    elif choice == 2:
        database = which_database()
        path = "./benchmark/" + database

        save_statistics(path, database)



def main():
    print("Multimedia Retrieval application")

    choice = menu()

    while choice != 0:

        if choice == 1:
            choice_1()

        elif choice == 2:
            choice_2()

        elif choice == 3:
            choice_3()

        elif choice == 4:
            choice_4()

        choice = menu()



if __name__ == "__main__":
    main()