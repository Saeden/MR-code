from Normalization import compute_pca, get_barycenter
from Mesh_Reading import *
from utils import *
import numpy as np
import csv
import random
from math import sqrt
from time import time


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


def compute_d3(mesh):

    t1 = time()

    n = 1000000
    vertices = np.asarray(mesh.vertices)
    num_of_vertices = len(vertices)
    k = int(n ** (1/3))
    result = np.zeros(k ** 3)
    index = 0

    for i in range(k):
        print("Iteration:", i + 1, "/", k)
        vi = int(num_of_vertices * random.random())
        p1 = vertices[vi]

        for j in range(k):
            vj = int(num_of_vertices * random.random())
            while vi == vj:
                vj = int(num_of_vertices * random.random())
            p2 = vertices[vj]

            for l in range(k):
                vl = int(num_of_vertices * random.random())
                while vi == vl or vj == vl:
                    vl = int(num_of_vertices * random.random())
                p3 = vertices[vl]

                area = 0.5*(np.linalg.norm(np.cross((p2-p1), (p3-p1))))
                result[index] = area
                index += 1
    t2 = time()
    print("Time:", t2-t1)
    return result


def compute_one_local_feature(mesh, file_name, feature):
    """
    This function compute a single local features - at user choice - of a single shape.
    :param mesh: the mesh whose local features have to be computed.
    :param feature: available modes -> 'a3', 'd1', 'd2', 'd3' or 'd4'.
    :return: a numpy array with the computed local features.
    """

    sample_count = 1000000

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
                while vi == vj:
                    vj = int(num_of_vertices * random.random())
                p2 = vertices[vj]

                if feature == 'd2':
                    distance = distance_between_2_points(p1, p2)
                    result[index] = distance
                    index += 1

                if sample_points > 2:
                    for k in range(root):
                        vk = int(num_of_vertices * random.random())
                        while vi == vk or vj == vk:
                            vk = int(num_of_vertices * random.random())
                        p3 = vertices[vk]

                        if feature == 'd3':
                            area = 0.5*(np.linalg.norm(np.cross((p2-p1), (p3-p1))))
                            result[index] = area
                            index += 1

                        elif feature == 'a3':
                            angle = compute_angle((p2-p1), (p3-p2))
                            result[index] = angle
                            index += 1

                        if sample_points > 3:
                            for l in range(root):
                                vl = int(num_of_vertices * random.random())
                                while vi == vl or vj == vl or vk == vl:
                                    vl = int(num_of_vertices * random.random())
                                p4 = vertices[vl]

                                volume = abs(volume_tetrahedron(p1, p2, p3, p4))
                                result[index] = volume
                                index += 1

    return result


"""def compute_one_local_feature_ok(mesh, file_name):

    t1 = time()

    root = 1000

    # initialization
    vertices = np.asarray(mesh.vertices)
    num_of_vertices = len(vertices)
    barycenter = get_barycenter(mesh)

    list_of_vertices = np.asarray(range(0, num_of_vertices))

    result_a3 = np.zeros(root*1000)
    result_d1 = np.zeros(root)
    result_d2 = np.zeros(root*10)
    result_d3 = np.zeros(root*1000)
    result_d4 = np.zeros(root*1000)

    index_a3 = 0
    index_d1 = 0
    index_d2 = 0
    index_d3 = 0
    index_d4 = 0

    # start of the algorithm
    for i in range(root):
        print("File:", file_name + ',', "iteration:", i+1, "/", root)
        vi = random.choice(list_of_vertices)
        p1 = vertices[vi]

        distance = distance_between_2_points(p1, barycenter)
        result_d1[index_d1] = distance
        index_d1 += 1

        if index_d1 % (root/10) == 0:
            for j in range(root):
                vj = random.choice(list_of_vertices)
                while vi == vj:
                    vj = random.choice(list_of_vertices)
                p2 = vertices[vj]

                distance = distance_between_2_points(p1, p2)
                result_d2[index_d2] = distance
                index_d2 += 1


                if index_d2 % (root/100) == 0:
                    for k in range(root):
                        vk = random.choice(list_of_vertices)
                        while vi == vk or vj == vk:
                            vk = random.choice(list_of_vertices)
                        p3 = vertices[vk]

                        area = 0.5*(np.linalg.norm(np.cross((p2-p1), (p3-p1))))
                        result_d3[index_d3] = area
                        index_d3 += 1

                        angle = compute_angle((p2-p1), (p3-p2))
                        result_a3[index_a3] = angle
                        index_a3 += 1

                        if index_d3 % (root) == 0:
                            for l in range(root):
                                vl = random.choice(list_of_vertices)
                                while vi == vl or vj == vl or vk == vl:
                                    vl = random.choice(list_of_vertices)
                                p4 = vertices[vl]

                                volume = abs(volume_tetrahedron(p1, p2, p3, p4))
                                result_d4[index_d4] = volume
                                index_d4 += 1

    t2 = time()

    print("Time:", t2 - t1)

    return result_a3, result_d1, result_d2, result_d3, result_d4"""



def compute_all_local_features(mesh, file_name):

    local_features = {}

    t1 = time()

    a3_raw = compute_one_local_feature(mesh, file_name, feature='a3')
    d1_raw = compute_one_local_feature(mesh, file_name, feature='d1')
    d2_raw = compute_one_local_feature(mesh, file_name, feature='d2')
    d3_raw = compute_one_local_feature(mesh, file_name, feature='d3')
    d4_raw = compute_one_local_feature(mesh, file_name, feature='d4')

    t2 = time()

    print(t2-t1)

    number_of_bins = 10

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

    output = []
    database = which_database()

    db_path = "./benchmark/" + database

    counter = 0

    for (root, dirs, files) in os.walk(db_path):

        for filename in files:

            if filename.endswith(".off") or filename.endswith(".ply"):

                filepath = root+'/'+filename
                file_name = filename[:filename.rfind('.')]

                mesh = load_mesh(filepath)

                all_features = compute_all_features_one_shape(mesh, file_name)
                counter += 1

                print("Number of shapes processed:", counter, "/ 1793")

                output.append(all_features)

    fieldnames = [i for i in all_features]

    filename = 'all_features.csv'

    with open(filename, 'w', encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output)


def distance_between_2_points(p1, p2):

    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)


def compute_angle(v1, v2):
    """
    Computes the angle between two vectors in radians.
    """
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
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



def try_():

    v = 1000
    iteration_i = 0
    iteration_j = 0
    iteration_k = 0
    iteration_l = 0

    for i in range(v):
        iteration_i += 1

        if iteration_i % (v/10) == 0:
            for j in range(v):
                iteration_j += 1

                if iteration_j % (v/100) == 0:
                    for k in range(v):
                        iteration_k += 1

                        if iteration_k % (v/4) == 0:
                            for l in range(v):
                                iteration_l += 1

    return iteration_i, iteration_j, iteration_k, iteration_l


def try_random():

    number = int(1450 * random.random())

    print(number)




mesh = load_mesh("./benchmark/db_ref_normalised/0/m99/m99.off")
#result_a3, result_d1, result_d2, result_d3, result_d4 = compute_one_local_feature_ok(mesh, "m99")
#feat = compute_all_features_one_shape(mesh, "m99")
#print(feat)

result = compute_all_local_features(mesh, "m99")

print(result)



