"""
To make the application works, install:

annoy==1.17.0
colorcet==2.0.6
matplotlib==3.4.3
mplcursors==0.5
numpy==1.21.2
open3d==0.13.0
pandas==1.3.3
pyglet==1.5.21
scikit-learn==0.24.2
scipy==1.7.1
seaborn==0.11.2
setuptools==58.1.0
trimesh==3.9.30

It's best to use Python 3.7 so to not have problems with the open3d library
(doesn't accept yet latest versions, like 3.8).

"""
from Mesh_Reading import load_mesh, view_mesh
from Statistics import *
from utils import get_path, get_read_params, which_database
from Mesh_refining import refine_single_mesh, refine_all_meshes
from Normalization import *
from Query_Meshes import query_interface, compute_all_distances
from Compute_Features import *
from Standardise_Features import *
from Dimensionality_Reduction import dimensionality_reduction
from Evaluation import *
from setuptools import setup, find_packages


def menu():

    number_of_choices = 8
    possible_choices = [str(i) for i in range(number_of_choices)]
    print("\n\n1) Read and visualize a single shape")
    print("2) Re-meshing operations")
    print("3) Normalizing operations")
    print("4) Feature extraction operations")
    print("5) Compute and save custom distance from each shape to another in a database")
    print("6) Evaluation operations")
    print("7) View/save statistics")
    print("\nPress 0 to go back.")
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
    print("\n1) Re-mesh one shape and visualize the result")
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

    number_of_choices = 8
    possible_choices = [str(i) for i in range(number_of_choices)]
    print("\n\n1) Extract and view only elementary global features on one shape")
    print("2) Extract and view only histogram global features on one shape")
    print("3) Extract and view all the features on one shape")
    print("4) Extract and save all the features of all shapes in a database")
    print("5) Standardise, weight and view the features of a shape")
    print("6) Standardise, weight and save all the features of all shapes in a database")
    print("7) Perform and save a volume differences analysis (fill_holes vs no fill_holes)")

    print("\nPress 0 to go back.")
    choice = input("\nChoice: ")

    while choice not in possible_choices:
        print("\nError! Invalid choice")
        choice = input("\nChoice: ")

    choice = int(choice)

    if choice == 1:

        print("\nOn which shape you want to extract the elementary global features?")

        path = get_path()
        mesh = load_mesh(path)
        global_features = compute_global_features(mesh)

        for feature, value in global_features.items():
            print(feature + ': ' + str(value))

    elif choice == 2:

        print("\nOn which shape you want to extract the histogram global features?")

        path = get_path()
        mesh = load_mesh(path)
        file_name = path[:path.rfind('/')]
        global_features = compute_all_local_features(mesh, file_name)

        for feature, value in global_features.items():
            print(feature + ': ' + str(value))

    elif choice == 3:

        print("\nOn which shape you want to extract all the features?")

        path = get_path()
        mesh = load_mesh(path)
        if '/' in path:
            file_name = path[(path.rfind('/')+1):path.rfind('.')]
        else:
            file_name = path[(path.rfind('\\') + 1):path.rfind('.')]
        global_features = compute_all_features_one_shape(mesh, file_name)

        for feature, value in global_features.items():
            print(feature + ': ' + str(value))

    elif choice == 4:

        compute_all_features_database()

    elif choice == 5:

        print("\nOn which shape you want to standardise, weight and view its features?")

        shape = input("\nInsert the shape number (e.g. m99, or m1234, ...): ")

        all_features = pd.read_csv('all_features.csv', header=0)
        shape_features = all_features.loc[all_features['file_name'] == shape].to_dict('records')[0]

        standardised_features = normalise_feat(shape_features, norm_param_path="./normalisation_parameters.csv", bin_number=15)

        for feature, value in standardised_features.items():
            print(feature + ': ' + str(value))

    elif choice == 6:

        feat_path = "./all_features.csv"
        normalise_all_feats(feat_path, bin_number=15, save_feats=True, save_params=True)

    elif choice == 7:

        export_volume_differences()


def choice_5():

    compute_all_distances(norm_params_path="./normalisation_parameters.csv", bin_number=15, save=True)


def choice_6():

    number_of_choices = 5
    possible_choices = [str(i) for i in range(number_of_choices)]
    print("\n\n1) Evaluate the performance (metrics) of our own distance function")
    print("2) Evaluate the performance (metrics) of ANN")
    print("3) Compute the ROC curve and AUC for each class separately")
    print("4) Compute the overall ROC curve and AUC")
    print("\nPress 0 to go back.")
    choice = input("\nChoice: ")

    while choice not in possible_choices:
        print("\nError! Invalid choice")
        choice = input("\nChoice: ")

    choice = int(choice)

    if choice == 1:
        evaluate_db()

    elif choice == 2:
        evaluate_ann()

    elif choice == 3:
        print("To be implemented")

    elif choice == 4:
        compute_roc_curve()


def choice_7():

    number_of_choices = 4
    possible_choices = [str(i) for i in range(number_of_choices)]
    print("\n\n1) Show statistics of one shape")
    print("2) Save statistics of a database")
    print("3) Plot the dimensionality reduction scatterplot")
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

    elif choice == 3:
        dimensionality_reduction("./normalised_features.csv")



def main():
    print("Multimedia Retrieval application")

    number_of_choices = 2
    possible_choices = [str(i) for i in range(number_of_choices + 1)]
    print("\n\n1) Preprocess mesh menu")
    print("2) Query mesh menu")

    print("\nPress 0 to exit.")
    choice = input("Choice: ")

    while choice not in possible_choices:
        print("Error! Invalid choice")
        choice = input("\nChoice: ")

    try:
        choice = int(choice)
    except ValueError:
        print("Error! Please enter a number.")

    while choice != 0:

        if choice == 1:

            choice1 = menu()

            while choice1 != 0:

                if choice1 == 1:
                    choice_1()

                elif choice1 == 2:
                    choice_2()

                elif choice1 == 3:
                    choice_3()

                elif choice1 == 4:
                    choice_4()

                elif choice == 5:
                    choice_5()

                elif choice == 6:
                    choice_6()

                elif choice1 == 7:
                    choice_7()

                choice1 = menu()

        elif choice == 2:
            query_interface()

        number_of_choices = 2
        possible_choices = [str(i) for i in range(number_of_choices + 1)]
        print("\n\n1) Preprocess mesh menu")
        print("2) Query mesh menu")

        print("\nPress 0 to exit.")
        choice = input("Choice: ")

        while choice not in possible_choices:
            print("Error! Invalid choice")
            choice = input("\nChoice: ")

        try:
            choice = int(choice)
        except ValueError:
            print("Error! Please enter a number.")


if __name__ == "__main__":
    packages = find_packages()
    print(packages)
    main()
