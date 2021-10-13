import os

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

    databases = []

    for (root, dirs, file) in os.walk("./benchmark"):
        for name in dirs:
            if name.startswith("db"):
                databases.append(name)

    print("Which database you want to use?")

    for i in range(len(databases)):
        print(str(i+1) + ")", databases[i])

    choice = int(input("Choice: "))

    return databases[choice-1]


def get_path():

    shape = input("\nInsert the shape number (e.g. m99, or m1234, ...): ")

    database = which_database()

    shape_number = int(shape[1:])

    shape_folder = str(int(shape_number/100))

    path = "./benchmark/" + database + "/" + shape_folder + "/" + shape + "/" + shape + ".off"

    print(path)

    return path


def get_read_params():

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

