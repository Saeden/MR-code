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


