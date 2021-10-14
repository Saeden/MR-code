import os
import shutil
import open3d as o3d


def read_classification_data(filepath, classification_dict=None):
    """
    :param filepath: path where the classification data are contained.
    :param classification_dict: a dictionary of the form {shape_num (int): 'classification'}.
    It can be passed empty, or with some data already inside (e.g. call this function twice,
    the first time on the training sample and the second on the test sample.
    :return: a dictionary of the form {shape_num (int): 'classification'}.
    """

    if classification_dict == None:
        classification_dict = {}

    file1 = open(filepath, 'r')
    lines = file1.readlines()

    object_name = ""

    for line in lines:
        if line == '\n':
            pass

        elif not line.strip().isdigit():
            raw_name = line.strip()
            last_space_index = raw_name.rfind(" ")
            removed_number_name = raw_name[:last_space_index]
            if removed_number_name[len(removed_number_name) - 1] == '0':
                removed_number_name = removed_number_name[:(len(removed_number_name) - 2)]
            object_name = removed_number_name

        else:
            classification_dict[int(line.strip())] = object_name

    return classification_dict


def get_complete_classification():
    """
    This function read from the coarse1 data, and will return a dictionary with all
    the shapes classified as described in the coarse1 data.
    :return: a dictionary of the form {shape_num (int): 'classification'}.
    """

    filepath_train = "./benchmark/classification/v1/coarse1/coarse1Train.cla"
    filepath_test = "./benchmark/classification/v1/coarse1/coarse1Test.cla"

    classification_dict_train = read_classification_data(filepath_train)

    classification_dict_complete = read_classification_data(filepath_test, classification_dict_train)

    return classification_dict_complete


def which_database():
    """
    This function allows to select on which database, among the available ones,
    a certain operation should be performed.
    :return: the string containing the name of the chosen database.
    """

    databases = []

    for (root, dirs, file) in os.walk("./benchmark"):
        for name in dirs:
            if name.startswith("db"):
                databases.append(name)

    number_of_choices = len(databases)
    possible_choices = [str(i) for i in range(number_of_choices+1)]
    possible_choices.remove("0")
    print("Which database you want to use?")

    for i in range(len(databases)):
        print(str(i+1) + ")", databases[i])

    choice = input("Choice: ")

    while choice not in possible_choices:
        print("\nError! Invalid choice")
        choice = input("\nChoice: ")

    choice = int(choice)

    return databases[choice-1]


def get_path():
    """
    This function will construct the path of a shape that has to be opened.
    It asks the name of the shape, and in which database it is contained.
    :return: the path of the shape to be opened.
    """

    shape = input("\nInsert the shape number (e.g. m99, or m1234, ...): ")

    database = which_database()

    shape_number = int(shape[1:])

    shape_folder = str(int(shape_number/100))

    path = "./benchmark/" + database + "/" + shape_folder + "/" + shape + "/" + shape + ".off"

    return path


def get_read_params():
    """
    This function allows the choice of displaying the x, y, z coordinates from the origin and the
    axis-align bounding box, when a shape is goign to be displayed.
    :return: the decision of the user about the coordinates and the axis-aligned bounding box.
    """

    draw_coordinates = int(input("\nDo you want to show the axis coordinates? (1 for yes / 0 for no): "))
    if draw_coordinates == 1:
        draw_coordinates = True
    else:
        draw_coordinates = False

    aabbox = int(input("Do you want to show the axis-aligned bounding box? (1 for yes / 0 for no): "))
    if aabbox == 1:
        aabbox = True
    else:
        aabbox = False

    return draw_coordinates, aabbox


def create_new_db(path):
    """
    This function creates an empty database with 19 folders in the specified path.
    :param path: the path where the database should be created.
    """

    if os.path.exists(path):
        shutil.rmtree(path)

    # make the directories that will contain the new db:
    os.mkdir(path)

    directories = range(19)

    for i in directories:
        new_path = path + "/" + str(i)
        os.mkdir(new_path)


def save_shape(filename, path, mesh):
    """
    This function saves the shape 'mesh' called 'filename' in the directory given by 'path'.
    :param filename: the name of the shape to save.
    :param path: the path to the database where we want to save the shape.
    :param mesh: the open3d object containing the shape that has to be saved.
    """

    file_code = filename[:-4]
    shape_number = int(file_code[1:])
    shape_folder = str(int(shape_number / 100))

    new_root = path + '/' + shape_folder + '/' + file_code

    os.mkdir(new_root)

    new_filepath = (new_root + "/" + filename)

    o3d.io.write_triangle_mesh(new_filepath, mesh, write_vertex_normals=False)

