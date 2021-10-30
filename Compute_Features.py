from Normalization import compute_pca, get_barycenter
from Mesh_Reading import *
from utils import *
import numpy as np
import csv
import random
from math import sqrt
import pandas as pd


def volume_tetrahedron(p1, p2, p3, center):

    return (1/6) * np.dot(np.cross((p1 - center), (p2 - center)), (p3 - center))


def get_volume(mesh):

    volume = 0
    center = get_barycenter(mesh)

    for face in np.asarray(mesh.triangles):

        p1 = np.asarray(mesh.vertices)[face[0]]
        p2 = np.asarray(mesh.vertices)[face[1]]
        p3 = np.asarray(mesh.vertices)[face[2]]

        volume += volume_tetrahedron(p1, p2, p3, center)

    return abs(volume)


def compute_global_features(mesh):

    global_features = {}

    area = mesh.get_surface_area()
    volume = get_volume(mesh)

    compactness = (area**3)/(36*np.pi*(volume**2))
    sphericity = 1/compactness

    # diameter as the distance of the max_bound vertex and min_bound vertex of the shape.
    diameter = distance_between_2_points(mesh.get_max_bound(), mesh.get_min_bound())

    aabbox_volume = o3d.geometry.AxisAlignedBoundingBox.get_axis_aligned_bounding_box(mesh).volume()
    rectangularity = volume / aabbox_volume

    eigenvectors, eigenvalues = compute_pca(mesh)
    eccentricity = abs(max(eigenvalues))/abs(min(eigenvalues))

    global_features['area'] = area
    global_features['volume'] = volume
    global_features['compactness'] = compactness
    global_features['sphericity'] = sphericity
    global_features['diameter'] = diameter
    global_features['aabbox_volume'] = aabbox_volume
    global_features['rectangularity'] = rectangularity
    global_features['eccentricity'] = eccentricity

    return global_features


def compute_one_local_feature(mesh, file_name, feature):
    """
    This function compute a single local features - at user choice - of a single shape.
    :param mesh: the mesh whose local features have to be computed.
    :param feature: available modes -> 'a3', 'd1', 'd2', 'd3' or 'd4'.
    :return: a numpy array with the computed local features.
    """

    sample_count = 100000

    if feature == 'a3' or feature == 'd3':
        sample_points = 3

    if feature == 'd1':
        sample_points = 1
        barycenter = get_barycenter(mesh)

    if feature == 'd2':
        sample_points = 2

    if feature == 'd4':
        sample_points = 4

    # initialization
    vertices = np.asarray(mesh.vertices)
    num_of_vertices = len(vertices)

    if num_of_vertices**sample_points > sample_count:
        root = int(sample_count ** (1/sample_points))
    else:
        root = num_of_vertices

    result = np.zeros(root**sample_points)
    index = 0

    # start of the algorithm
    for i in range(root):
        print("File:", file_name + ',', "feature:", feature + ',', "iteration:", i+1, "/", root)
        vi = int(num_of_vertices * random.random())
        p1 = vertices[vi]

        if feature == 'd1':
            distance = distance_between_2_points(p1, barycenter)
            result[index] = distance
            index += 1

        if sample_points > 1:
            for j in range(root):
                vj = int(num_of_vertices * random.random())
                p2 = vertices[vj]
                while (p1 == p2).all():
                    vj = int(num_of_vertices * random.random())
                    p2 = vertices[vj]

                if feature == 'd2':
                    distance = distance_between_2_points(p1, p2)
                    result[index] = distance
                    index += 1

                if sample_points > 2:
                    for k in range(root):
                        vk = int(num_of_vertices * random.random())
                        p3 = vertices[vk]
                        while (p1 == p3).all() or (p2 == p3).all():
                            vk = int(num_of_vertices * random.random())
                            p3 = vertices[vk]

                        if feature == 'd3':
                            area = sqrt(0.5*(np.linalg.norm(np.cross((p2-p1), (p3-p1)))))
                            result[index] = area
                            index += 1

                        elif feature == 'a3':

                            angle = compute_angle((p2-p1), (p3-p2))
                            result[index] = angle
                            index += 1

                        if sample_points > 3:
                            for l in range(root):
                                vl = int(num_of_vertices * random.random())
                                p4 = vertices[vl]
                                while (p1 == p4).all() or (p2 == p4).all() or (p3 == p4).all():
                                    vl = int(num_of_vertices * random.random())
                                    p4 = vertices[vl]

                                volume = np.cbrt(abs(volume_tetrahedron(p1, p2, p3, p4)))
                                result[index] = volume
                                index += 1

    return result


def compute_all_local_features(mesh, file_name):

    local_features = {}

    a3_raw = compute_one_local_feature(mesh, file_name, feature='a3')
    d1_raw = compute_one_local_feature(mesh, file_name, feature='d1')
    d2_raw = compute_one_local_feature(mesh, file_name, feature='d2')
    d3_raw = compute_one_local_feature(mesh, file_name, feature='d3')
    d4_raw = compute_one_local_feature(mesh, file_name, feature='d4')

    number_of_bins = 15

    a3, a3_bin = np.histogram(a3_raw, bins=number_of_bins)
    d1, d1_bin = np.histogram(d1_raw, bins=number_of_bins)
    d2, d2_bin = np.histogram(d2_raw, bins=number_of_bins)
    d3, d3_bin = np.histogram(d3_raw, bins=number_of_bins)
    d4, d4_bin = np.histogram(d4_raw, bins=number_of_bins)

    # filling the dictionary with the new features
    for i in range(number_of_bins):
        local_features['a3_' + str(i+1)] = a3[i]/sum(a3)
    for i in range(number_of_bins):
        local_features['a3_range_' + str(i+1)] = a3_bin[i]

    for i in range(number_of_bins):
        local_features['d1_' + str(i+1)] = d1[i]/sum(d1)
    for i in range(number_of_bins):
        local_features['d1_range_' + str(i+1)] = d1_bin[i]

    for i in range(number_of_bins):
        local_features['d2_' + str(i+1)] = d2[i]/sum(d2)
    for i in range(number_of_bins):
        local_features['d2_range_' + str(i+1)] = d2_bin[i]

    for i in range(number_of_bins):
        local_features['d3_' + str(i+1)] = d3[i]/sum(d3)
    for i in range(number_of_bins):
        local_features['d3_range_' + str(i+1)] = d3_bin[i]

    for i in range(number_of_bins):
        local_features['d4_' + str(i+1)] = d4[i]/sum(d4)
    for i in range(number_of_bins):
        local_features['d4_range_' + str(i+1)] = d4_bin[i]

    return local_features


def compute_all_features_one_shape(mesh, file_name):

    mesh_data = {}

    mesh_data['file_name'] = file_name
    mesh_data['shape_number'] = int(file_name[1:])

    global_features = compute_global_features(mesh)
    local_features = compute_all_local_features(mesh, file_name)

    # merge the dictionaries
    all_features = {**mesh_data, **global_features, **local_features}

    return all_features


def compute_all_features_database():

    database = which_database()

    db_path = "./benchmark/" + database

    counter = 0

    features_filename = 'all_features.csv'

    if not os.path.exists(features_filename):
        with open(features_filename, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            column_names = get_column_names()
            writer.writerow(column_names)

    for (root, dirs, files) in os.walk(db_path):

        for filename in files:

            file_name = filename[:filename.rfind('.')]
            df = pd.read_csv(features_filename)
            processed = df['file_name'].tolist()

            if (filename.endswith(".off") or filename.endswith(".ply")) and file_name not in processed:

                filepath = root+'/'+filename

                mesh = load_mesh(filepath)

                all_features = compute_all_features_one_shape(mesh, file_name)

                df = df.append(all_features, ignore_index=True)

            counter += 1

            if os.path.exists(features_filename):
                os.remove(features_filename)
            df.to_csv(features_filename, index=False)

            print("Number of shapes processed:", counter, "/ 1793")


def distance_between_2_points(p1, p2):

    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)


def compute_angle(v1, v2):
    """
    The computed angle is in radians.
    """
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    if cos_angle > 1:
        cos_angle = 1

    if cos_angle < -1:
        cos_angle = -1

    angle = np.arccos(cos_angle)

    return angle


def export_volume_differences():

    output = []
    database = which_database()

    db_path = "./benchmark/" + database

    for (root, dirs, files) in os.walk(db_path):

        for filename in files:

            if filename.endswith(".off") or filename.endswith(".ply"):

                data_obtained = {}

                filepath = root+'/'+filename

                tm_mesh = load_mesh_trimesh(filepath)
                o3d_mesh = load_mesh(filepath)

                is_original_watertight = tm_mesh.is_watertight

                original_tm_volume = tm_mesh.volume

                tm_mesh.fix_normals()
                tm_mesh.remove_duplicate_faces()
                tm_mesh.fill_holes()

                is_tm_watertight = tm_mesh.is_watertight

                tm_volume = tm_mesh.volume
                o3d_volume = get_volume(o3d_mesh)

                file_name = filename[:filename.rfind('.')]
                data_obtained['file_name'] = file_name
                data_obtained['shape_number'] = int(file_name[1:])
                data_obtained['watertight_before'] = is_original_watertight
                data_obtained['watertight_after'] = is_tm_watertight
                data_obtained['tm_volume_before'] = original_tm_volume
                data_obtained['tm_volume_after'] = tm_volume
                data_obtained['o3d_volume'] = o3d_volume

                output.append(data_obtained)

    fieldnames = [i for i in data_obtained]

    filename = 'volume_differences.csv'

    with open(filename, 'w', encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output)


def export_features_one_shape(features_dict, filename):

    output = [features_dict]
    fieldnames = [i for i in features_dict]

    filename = filename + '_features.csv'

    with open(filename, 'w', encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output)


def get_column_names():

    column_names = ["file_name", "shape_number", "area", "volume", "compactness", "sphericity", "diameter", "aabbox_volume", "rectangularity", "eccentricity",
                    "a3_1", "a3_2", "a3_3", "a3_4", "a3_5", "a3_6", "a3_7", "a3_8", "a3_9", "a3_10", "a3_11", "a3_12", "a3_13", "a3_14", "a3_15",
                    "a3_range_1", "a3_range_2", "a3_range_3", "a3_range_4", "a3_range_5", "a3_range_6", "a3_range_7", "a3_range_8", "a3_range_9", "a3_range_10", "a3_range_11", "a3_range_12", "a3_range_13", "a3_range_14", "a3_range_15",
                    "d1_1", "d1_2", "d1_3", "d1_4", "d1_5", "d1_6", "d1_7", "d1_8", "d1_9", "d1_10", "d1_11", "d1_12", "d1_13", "d1_14", "d1_15",
                    "d1_range_1", "d1_range_2", "d1_range_3", "d1_range_4", "d1_range_5", "d1_range_6", "d1_range_7", "d1_range_8", "d1_range_9", "d1_range_10", "d1_range_11", "d1_range_12", "d1_range_13", "d1_range_14", "d1_range_15",
                    "d2_1", "d2_2", "d2_3", "d2_4", "d2_5", "d2_6", "d2_7", "d2_8", "d2_9", "d2_10", "d2_11", "d2_12", "d2_13", "d2_14", "d2_15",
                    "d2_range_1", "d2_range_2", "d2_range_3", "d2_range_4", "d2_range_5", "d2_range_6", "d2_range_7", "d2_range_8", "d2_range_9", "d2_range_10", "d2_range_11", "d2_range_12", "d2_range_13", "d2_range_14", "d2_range_15",
                    "d3_1", "d3_2", "d3_3", "d3_4", "d3_5", "d3_6", "d3_7", "d3_8", "d3_9", "d3_10", "d3_11", "d3_12", "d3_13", "d3_14", "d3_15",
                    "d3_range_1", "d3_range_2", "d3_range_3", "d3_range_4", "d3_range_5", "d3_range_6", "d3_range_7", "d3_range_8", "d3_range_9", "d3_range_10", "d3_range_11", "d3_range_12", "d3_range_13", "d3_range_14", "d3_range_15",
                    "d4_1", "d4_2", "d4_3", "d4_4", "d4_5", "d4_6", "d4_7", "d4_8", "d4_9", "d4_10", "d4_11", "d4_12", "d4_13", "d4_14", "d4_15",
                    "d4_range_1", "d4_range_2", "d4_range_3", "d4_range_4", "d4_range_5", "d4_range_6", "d4_range_7", "d4_range_8", "d4_range_9", "d4_range_10", "d4_range_11", "d4_range_12", "d4_range_13", "d4_range_14", "d4_range_15"]

    return column_names
