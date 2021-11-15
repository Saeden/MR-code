import pandas as pd
import csv
import numpy as np
import open3d as o3d
from scipy.stats import wasserstein_distance as emd
from scipy.spatial.distance import euclidean
from Mesh_Reading import load_mesh
from utils import get_path_from_shape
from Compute_Features import compute_all_features_one_shape
from Normalization import normalise_mesh_step2
from Mesh_refining import refine_single_mesh
from Standardise_Features import *
import os.path
from ANN import ann


def compute_all_distances(norm_params_path="./normalisation_parameters.csv", bin_number=15, save=False):

    global_feats = ['area', 'volume', 'compactness', 'sphericity', 'diameter', 'aabbox_volume', 'rectangularity',
                    'eccentricity']
    hist_feats = ['a3_', 'd1_', 'd2_', 'd3_', 'd4_']
    output = []

    if not os.path.isfile("./normalised_features.csv"):
        print("\nThe features have not been normalised...")
        print("Normalising now.\n")
        normed_feats, norm_params = normalise_all_feats("./all_features.csv", save_feats=True)

    else:
        with open("./normalised_features.csv") as f:
            normed_feats = [{k: v for k, v in row.items()}
                 for row in csv.DictReader(f)]
        with open(norm_params_path) as f:
            norm_params = [{k: v for k, v in row.items()}
                           for row in csv.DictReader(f)][0]

    index = 0
    printed = False
    print("\nCalculating distances now.")
    for feat1 in normed_feats:
        index += 1
        dist_to_meshes = {'file_name': feat1['file_name'], 'shape_number': feat1['shape_number']}
        completion = int(((index) / len(normed_feats)) * 100)
        if completion % 5 == 0 and not printed:
            print(f"Found {completion}% of distances.")
            printed = True
        if completion % 5 == 1:
                printed = False
        for feat2 in normed_feats:
            global_dist = 0
            hist_dist = []
            """if feat1['file_name'] == feat2['file_name']:
                dist_to_meshes[f"dist_to_{feat2['file_name']}"] = 0
            else:"""
            for gf in global_feats:
                global_dist += (float(feat1[gf])-float(feat2[gf]))**2
            global_dist = np.sqrt(global_dist)

            for hf in hist_feats:
                feat1_hist = [feat1[str(hf + str(i + 1))] for i in range(bin_number)]
                feat2_hist = [feat2[str(hf + str(i + 1))] for i in range(bin_number)]

                #avg = float(norm_params[str('avg_'+hf[:-1])])
                #std = float(norm_params[str('std_'+hf[:-1])])
                dist = emd(feat1_hist, feat2_hist)
                #dist = euclidean(feat1_hist, feat2_hist)

                normed_dist = dist#(dist-avg)/std

                hist_dist.append(abs(normed_dist))

            total_dist = (sum(hist_dist)+global_dist)/(len(hist_dist)+1)

            dist_to_meshes[f"dist_to_{feat2['file_name']}"] = total_dist

        output.append(dist_to_meshes)

    if save:
        fieldnames = output[0].keys()

        filename = 'distance_to_meshes.csv'

        with open(filename, 'w', encoding='UTF8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output)

    return output


def query_db_mesh(mesh_name, num_closest_meshes=10):

    if not os.path.isfile("./distance_to_meshes.csv"):
        print("\nThe distances have not been calculated yet...")
        compute_all_distances("./normalised_features.csv", save=True)
        distance_db = pd.read_csv("./distance_to_meshes.csv", header=0)
    else:
        distance_db = pd.read_csv("./distance_to_meshes.csv", header=0)

    closest_meshes = []
    mesh_distances = distance_db.loc[distance_db['file_name'] == mesh_name].to_dict(orient='records')[0]
    del mesh_distances['file_name']
    del mesh_distances['shape_number']
    #del mesh_distances['Class']
    sorted_mesh_dist = {key:val for key, val in sorted(mesh_distances.items(), key=lambda item: item[1])}
    for i in range(num_closest_meshes):
        closest_meshes.append((list(sorted_mesh_dist)[i+1][8:], list(sorted_mesh_dist.values())[i+1]))

    return closest_meshes, mesh_name


def query_db_mesh_fast(mesh_name, distance_db, num_closest_meshes=10):
    closest_meshes = []
    mesh_distances = distance_db.loc[distance_db['file_name'] == mesh_name].to_dict(orient='records')[0]
    del mesh_distances['file_name']
    del mesh_distances['shape_number']
    #del mesh_distances['Class']
    sorted_mesh_dist = {key:val for key, val in sorted(mesh_distances.items(), key=lambda item: item[1])}
    for i in range(num_closest_meshes):
        closest_meshes.append((list(sorted_mesh_dist)[i+1][8:], list(sorted_mesh_dist.values())[i+1]))

    return closest_meshes, mesh_name


# to be modified?
def compute_single_distance(mesh_feat, norm_params_path="./normalisation_parameters.csv",  normed_feats_path="./normalised_features.csv", bin_number=15):
    global_feats = ['area', 'volume', 'compactness', 'sphericity', 'diameter', 'aabbox_volume', 'rectangularity',
                    'eccentricity']
    hist_feats = ['a3_', 'd1_', 'd2_', 'd3_', 'd4_']

    if not os.path.isfile(normed_feats_path):
        print("\nThe features have not been normalised.")
        normed_feats, norm_params = normalise_all_feats("./all_features.csv")

    else:
        with open(normed_feats_path) as f:
            normed_feats = [{k: v for k, v in row.items()}
                            for row in csv.DictReader(f)]
        with open(norm_params_path) as f:
            norm_params = [{k: v for k, v in row.items()}
                           for row in csv.DictReader(f)][0]

    dist_to_meshes = {'file_name': mesh_feat['file_name'], 'shape_number': mesh_feat['shape_number']}
    index = 0
    printed = False
    for feat2 in normed_feats:
        index += 1
        completion = int(((index) / len(normed_feats)) * 100)
        if completion % 5 == 0 and not printed:
            print(f"Found {completion}% of distances.")
            printed = True
        if completion % 5 == 1:
            printed = False

        global_dist = 0
        hist_dist = []
        if mesh_feat['file_name'] == feat2['file_name']:
            dist_to_meshes[f"dist_to_{feat2['file_name']}"] = 0
        else:
            for gf in global_feats:
                global_dist += (float(mesh_feat[gf])-float(feat2[gf]))**2
            global_dist = np.sqrt(global_dist)

            for hf in hist_feats:
                mesh_feat_hist = [mesh_feat[str(hf+str(i+1))] for i in range(bin_number)]
                feat2_hist = [feat2[str(hf + str(i + 1))] for i in range(bin_number)]

                avg = float(norm_params[str('avg_' + hf[:-1])])
                std = float(norm_params[str('std_' + hf[:-1])])
                dist = emd(mesh_feat_hist, feat2_hist)

                normed_dist = (dist - avg) / std

                hist_dist.append(abs(normed_dist))

            total_dist = (sum(hist_dist)+global_dist)/len(hist_dist)+1

            dist_to_meshes[f"dist_to_{feat2['file_name']}"] = total_dist

    print(f"Finished calculating distances for mesh {mesh_feat['file_name']}")

    return dist_to_meshes


def query_new_mesh(mesh_path, num_closest_meshes=25):
    """This function loads a new mesh, normalises it and extracts its features. Then a query is performed on the db.
    :return: closest_meshes a list of tuples (mesh_name, distance) and mesh the newly loaded/normalised mesh"""

    print("Which method would you like to use? (ANN or dist)")
    choiceM = input("Method: ")

    if choiceM == "dist":
        mesh = load_mesh(mesh_path)
        print("\nRe-meshing the shape.")
        ref_mesh = refine_single_mesh(mesh)
        print("\nNormalising mesh.")
        norm_mesh = normalise_mesh_step2(ref_mesh)
        print("\nComputing features for new shape.")
        mesh_feats = compute_all_features_one_shape(norm_mesh, "m0000")
        print("\nNormalising features of new shape.")
        normed_feats = normalise_feat(mesh_feats)
        print("\nCalculating distances of new shape to the database.")

        mesh_distances = compute_single_distance(normed_feats, normed_feats_path="./normalised_features.csv")
        del mesh_distances['file_name']
        del mesh_distances['shape_number']
        # del mesh_distances['Class']
        closest_meshes = []
        sorted_mesh_dist = {key: val for key, val in sorted(mesh_distances.items(), key=lambda item: item[1])}
        for i in range(num_closest_meshes):
            closest_meshes.append((list(sorted_mesh_dist)[i + 1][8:], list(sorted_mesh_dist.values())[i + 1]))
    else:
        mesh_name = mesh_path
        closest_meshes, norm_mesh = ann(query_mesh=mesh_name, feature_list="./normalised_features.csv", num_of_trees=1000, top_k=num_closest_meshes, query=True)
        del closest_meshes[0]
        del closest_meshes[-1]

    print("\nThe results from the query are: ")
    print(closest_meshes)

    return closest_meshes, norm_mesh


def display_query(closest_meshes, qmesh_name=None, new_qmesh=None):
    """"Function to display the queried mesh and the results, expects a mesh name (m0-m1762) or a new mesh object,
    the new mesh object should be normalised. closest_meshes should be a list of tuples ('mesh_name', distance)
    which is the query result.
    """
    num_meshes = len(closest_meshes)
    database = "db_ref_normalised"
    mesh_objs = []
    if not qmesh_name == None:
        path = get_path_from_shape(qmesh_name, database)
        qmesh = load_mesh(path)
    else:
        qmesh = new_qmesh

    qmesh.compute_vertex_normals()
    mesh_objs.append(qmesh)

    for i in range(num_meshes):
        shape_name = closest_meshes[i][0]
        path = get_path_from_shape(shape_name, database)
        cmesh = load_mesh(path)
        #x_transl = (i%5+1)*1.1
        #z_transl = -int(i/5)
        cmesh = cmesh.translate(translation=[(i%5+1)*1.1, int(i/5), 0])
        cmesh.compute_vertex_normals()
        mesh_objs.append(cmesh)

    o3d.visualization.draw_geometries(mesh_objs)


def query_interface():
    print("\nQuerying meshes options:")
    print("1) Query a mesh from the database.")
    print("2) Load your own mesh to query.")
    print("0) Exit this menu")

    number_of_choices = 3
    possible_choices = [i for i in range(number_of_choices)]
    choice1 = ""
    while isinstance(choice1, str):
        choice1 = input("\nChoice: ")
        try:
            choice1 = int(choice1)
        except ValueError:
            print("Error! Please enter a number.")

    while choice1 not in possible_choices:
        print("\nError! Invalid choice")
        choice1 = int(input("\nChoice: "))

    while choice1 != 0:
        if choice1 == 1:
            print("\nWhich mesh would you like to query? (m0, m1...)")
            choiceQ = input("Mesh: ")
            print("How many results would you like to have returned?")
            choiceR = int(input("Number of results: "))
            print("Which method would you like to use? (ANN or dist)")
            choiceM = input("Method: ")
            if choiceM == "dist":
                closest_meshes, mesh_name = query_db_mesh(choiceQ, choiceR)
            else:
                mesh_name = choiceQ
                closest_meshes = ann(query_mesh=mesh_name, feature_list="./normalised_features.csv", num_of_trees=1000, top_k=choiceR)
                del closest_meshes[0]
                del closest_meshes[-1]
            print("\nThe results from the query are: ")
            print(closest_meshes)
            print("\nWould you like to visualise the results? (1 for yes/0 for no)")
            choiceRes = int(input("Choice: "))
            if choiceRes == 1:
                display_query(closest_meshes, qmesh_name=mesh_name)

        elif choice1 == 2:
            print("\nWhat is the path to the mesh you want to query?")
            path = input("Path: ")
            print("How many results would you like to have returned?")
            choiceR = int(input("Number of results: "))
            closest_meshes, new_mesh = query_new_mesh(path, choiceR)
            print("\nWould you like to visualise the results? (1 for yes/0 for no)")
            choiceRes = int(input("Choice: "))
            if choiceRes == 1:
                display_query(closest_meshes, new_qmesh=new_mesh)

        print("\nQuerying meshes options:")
        print("1) Query a mesh from the database.")
        print("2) Load your own mesh to query.")
        print("0) Exit this menu")

        number_of_choices = 3
        possible_choices = [i for i in range(number_of_choices)]
        choice1 = ""
        while isinstance(choice1, str):
            choice1 = input("\nChoice: ")
            try:
                choice1 = int(choice1)
            except ValueError:
                print("Error! Please enter a number.")

        while choice1 not in possible_choices:
            print("\nError! Invalid choice")
            choice1 = int(input("\nChoice: "))



def test():

    mesh_name = "m1000"
    feature_list = "./normalised_features.csv"

    shapes = ann(mesh_name, feature_list, num_of_trees=1000, search_k=-1, top_k=25)

    del shapes[0]  # deleting the first element, which is the query shape with distance 0, so to not display it

    display_query(shapes, qmesh_name=mesh_name)

#test()

query_interface()

#compute_all_distances(save=True)