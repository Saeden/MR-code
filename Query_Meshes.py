import pandas as pd
import csv
import numpy as np
import open3d as o3d
from scipy.stats import wasserstein_distance as emd
from Mesh_Reading import load_mesh
from utils import get_path_from_shape
from Compute_Features import compute_all_features_one_shape
from Normalization import normalise_mesh_step2


def normalise_all_feats(feat_path, bin_number=15, save_feats=False, save_params=True):
    feats = pd.read_csv(feat_path, header=0)
    norm_params = {}
    all_feats = []
    hist_feats = ['a3_', 'd1_', 'd2_', 'd3_', 'd4_']
    hist_dist = {'a3_':[], 'd1_':[], 'd2_':[], 'd3_':[], 'd4_':[]}

    avg_area = sum(feats['area'])/len(feats['area'])
    std_area = np.std(feats['area'])
    norm_params['avg_area'] = avg_area
    norm_params['std_area'] = std_area

    avg_volume = sum(feats['volume'])/len(feats['volume'])
    std_volume = np.std(feats['volume'])
    norm_params['avg_volume'] = avg_volume
    norm_params['std_volume'] = std_volume

    avg_compactness = sum(feats['compactness'])/len(feats['compactness'])
    std_compactness = np.std(feats['compactness'])
    norm_params['avg_compactness'] =avg_compactness
    norm_params['std_compactness'] =std_compactness

    avg_sphericity = sum(feats['sphericity'])/len(feats['sphericity'])
    std_sphericity = np.std(feats['sphericity'])
    norm_params['avg_sphericity'] =avg_sphericity
    norm_params['std_sphericity'] =std_sphericity

    avg_diameter = sum(feats['diameter'])/len(feats['diameter'])
    std_diameter = np.std(feats['diameter'])
    norm_params['avg_diameter'] =avg_diameter
    norm_params['std_diameter'] =std_diameter

    avg_aabb_vol = sum(feats['aabbox_volume'])/len(feats['aabbox_volume'])
    std_aabb_vol = np.std(feats['aabbox_volume'])
    norm_params['avg_aabb_vol'] =avg_aabb_vol
    norm_params['std_aabb_vol'] =std_aabb_vol

    avg_rect = sum(feats['rectangularity'])/len(feats['rectangularity'])
    std_rect = np.std(feats['rectangularity'])
    norm_params['avg_rect'] =avg_rect
    norm_params['std_rect'] =std_rect

    avg_eccent = sum(feats['eccentricity'])/len(feats['eccentricity'])
    std_eccent = np.std(feats['eccentricity'])
    norm_params['avg_eccent'] = avg_eccent
    norm_params['std_eccent'] = std_eccent

    for index1, row1 in feats.iterrows():
        for index2, row2 in feats.iterrows():
            for hf in hist_feats:
                feat1_hist = [row1[str(hf + str(i + 1))] for i in range(bin_number)]
                feat2_hist = [row2[str(hf + str(i + 1))] for i in range(bin_number)]

                hist_dist[hf].append(emd(feat1_hist, feat2_hist))

    norm_params['avg_a3'] = sum(hist_dist['a3_'])/len(hist_dist['a3_'])
    norm_params['std_a3'] = np.std(hist_dist['a3_'])

    norm_params['avg_d1'] = sum(hist_dist['d1_']) / len(hist_dist['d1_'])
    norm_params['std_d1'] = np.std(hist_dist['d1_'])

    norm_params['avg_d2'] = sum(hist_dist['d2_']) / len(hist_dist['d2_'])
    norm_params['std_d2'] = np.std(hist_dist['d2_'])

    norm_params['avg_d3'] = sum(hist_dist['d3_']) / len(hist_dist['d3_'])
    norm_params['std_d3'] = np.std(hist_dist['d3_'])

    norm_params['avg_d4'] = sum(hist_dist['d4_']) / len(hist_dist['d4_'])
    norm_params['std_d4'] = np.std(hist_dist['d4_'])

    for index, row in feats.iterrows():
        norm_feats = {}
        norm_feats['file_name'] = row['file_name']
        norm_feats['shape_number'] = row['shape_number']
        norm_feats['area'] = (row['area']-avg_area)/std_area
        norm_feats['volume'] = (row['volume']-avg_volume)/std_volume
        norm_feats['compactness'] = (row['compactness']-avg_compactness)/std_compactness
        norm_feats['sphericity'] = (row['sphericity']-avg_sphericity)/std_sphericity
        norm_feats['diameter'] = (row['diameter']-avg_diameter)/std_diameter
        norm_feats['aabbox_volume'] = (row['aabbox_volume']-avg_aabb_vol)/std_aabb_vol
        norm_feats['rectangularity'] = (row['rectangularity']-avg_rect)/std_rect
        norm_feats['eccentricity'] = (row['eccentricity']-avg_eccent)/std_eccent


        for i in range(bin_number):
            norm_feats[f"a3_{i + 1}"] = row[f"a3_{i + 1}"]
            norm_feats[f"d1_{i + 1}"] = row[f"d1_{i + 1}"]
            norm_feats[f"d2_{i + 1}"] = row[f"d2_{i + 1}"]
            norm_feats[f"d3_{i + 1}"] = row[f"d3_{i + 1}"]
            norm_feats[f"d4_{i + 1}"] = row[f"d4_{i + 1}"]

        all_feats.append(norm_feats)


    if save_params:
        fieldnames = [i for i in norm_params]

        filename = 'normalisation_parameters.csv'

        with open(filename, 'w', encoding='UTF8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows([norm_params])

    if save_feats:
        fieldnames = [i for i in norm_feats]

        filename = 'normalised_features.csv'

        with open(filename, 'w', encoding='UTF8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_feats)

    return all_feats, norm_params


def normalise_feat(feats, norm_param_path="./normalisation_parameters.csv", bin_number=15):

    with open(norm_param_path) as f:
        norm_params = [{k: v for k, v in row.items()}
                       for row in csv.DictReader(f)][0]

    norm_feats = {}
    norm_feats['file_name'] = feats['file_name']
    norm_feats['shape_number'] = feats['shape_number']
    norm_feats['area'] = (feats['area']-float(norm_params['avg_area']))/float(norm_params['std_area'])
    norm_feats['volume'] = (feats['volume']-float(norm_params['avg_volume']))/float(norm_params['std_volume'])
    norm_feats['compactness'] = (feats['compactness']-float(norm_params['avg_compactness']))/float(norm_params['std_compactness'])
    norm_feats['sphericity'] = (feats['sphericity']-float(norm_params['avg_sphericity']))/float(norm_params['std_sphericity'])
    norm_feats['diameter'] = (feats['diameter']-float(norm_params['avg_diameter']))/float(norm_params['std_diameter'])
    norm_feats['aabbox_volume'] = (feats['aabbox_volume']-float(norm_params['avg_aabb_vol']))/float(norm_params['std_aabb_vol'])
    norm_feats['rectangularity'] = (feats['rectangularity']-float(norm_params['avg_rect']))/float(norm_params['std_rect'])
    norm_feats['eccentricity'] = (feats['eccentricity']-float(norm_params['avg_eccent']))/float(norm_params['std_eccent'])

    for i in range(bin_number):
        norm_feats[f"a3_{i+1}"] = feats[f"a3_{i+1}"]
        norm_feats[f"d1_{i + 1}"] = feats[f"d1_{i + 1}"]
        norm_feats[f"d2_{i + 1}"] = feats[f"d2_{i + 1}"]
        norm_feats[f"d3_{i + 1}"] = feats[f"d3_{i + 1}"]
        norm_feats[f"d4_{i + 1}"] = feats[f"d4_{i + 1}"]




    return norm_feats


def compute_all_distances(normed_feats_path=None,norm_params_path = "./normalisation_parameters.csv", bin_number=15, save=False):
    global_feats = ['area', 'volume', 'compactness', 'sphericity', 'diameter', 'aabbox_volume', 'rectangularity',
                    'eccentricity']
    hist_feats = ['a3_', 'd1_', 'd2_', 'd3_', 'd4_']
    output = []

    if normed_feats_path == None:
        normed_feats, norm_params = normalise_all_feats("./all_features.csv")

    else:
        with open(normed_feats_path) as f:
            normed_feats = [{k: v for k, v in row.items()}
                 for row in csv.DictReader(f)]
        with open(norm_params_path) as f:
            norm_params = [{k: v for k, v in row.items()}
                           for row in csv.DictReader(f)][0]

    for feat1 in normed_feats:
        dist_to_meshes = {'file_name': feat1['file_name'], 'shape_number': feat1['shape_number']}
        for feat2 in normed_feats:
            global_dist = 0
            hist_dist = []
            if feat1['file_name'] == feat2['file_name']:
                dist_to_meshes[f"dist_to_{feat2['file_name']}"] = 0
            else:
                for gf in global_feats:
                    global_dist += (float(feat1[gf])-float(feat2[gf]))**2
                global_dist = np.sqrt(global_dist)

                for hf in hist_feats:
                    feat1_hist = [feat1[str(hf+str(i+1))] for i in range(bin_number)]
                    feat2_hist = [feat2[str(hf + str(i + 1))] for i in range(bin_number)]

                    avg = float(norm_params[str('avg_'+hf[:-1])])
                    std = float(norm_params[str('std_'+hf[:-1])])
                    dist = emd(feat1_hist, feat2_hist)

                    normed_dist = (dist-avg)/std

                    hist_dist.append(abs(normed_dist))

                total_dist = (sum(hist_dist)+global_dist)/len(hist_dist)+1

                dist_to_meshes[f"dist_to_{feat2['file_name']}"] = total_dist

        print(f"Finished calculating distances for mesh {feat1['shape_number']}")

        output.append(dist_to_meshes)

    if save:
        fieldnames = output[0].keys()

        filename = 'distance_to_meshes.csv'

        with open(filename, 'w', encoding='UTF8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output)

    return output


def compute_single_distance(mesh_feat, norm_params_path="./normalisation_parameters.csv",  normed_feats_path=None, bin_number=15):
    global_feats = ['area', 'volume', 'compactness', 'sphericity', 'diameter', 'aabbox_volume', 'rectangularity',
                    'eccentricity']
    hist_feats = ['a3_', 'd1_', 'd2_', 'd3_', 'd4_']

    if normed_feats_path == None:
        normed_feats, norm_params = normalise_all_feats("./all_features.csv")

    else:
        with open(normed_feats_path) as f:
            normed_feats = [{k: v for k, v in row.items()}
                            for row in csv.DictReader(f)]
        with open(norm_params_path) as f:
            norm_params = [{k: v for k, v in row.items()}
                           for row in csv.DictReader(f)][0]

    dist_to_meshes = {'file_name': mesh_feat['file_name'], 'shape_number': mesh_feat['shape_number']}
    for feat2 in normed_feats:
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


def query_db_mesh(mesh_name, num_closest_meshes=10, distance_db_path=None):
    if distance_db_path == None:
        distance_db = compute_all_distances("./normalised_features.csv")

    else:
        distance_db = pd.read_csv(distance_db_path, header=0)

    closest_meshes = []
    mesh_distances = distance_db.loc[distance_db['file_name'] == mesh_name].to_dict(orient='records')[0]
    del mesh_distances['file_name']
    del mesh_distances['shape_number']
    sorted_mesh_dist = {key:val for key, val in sorted(mesh_distances.items(), key=lambda item: item[1])}
    for i in range(num_closest_meshes):
        closest_meshes.append((list(sorted_mesh_dist)[i+1][8:], list(sorted_mesh_dist.values())[i+1]))

    return closest_meshes, mesh_name

def query_new_mesh(mesh_path, mesh_name, num_closest_meshes=10):
    """This function loads a new mesh, normalises it and extracts its features. Then a query is performed on the db.
    :return: closest_meshes a list of tuples (mesh_name, distance) and mesh the newly loaded/normalised mesh"""
    mesh = load_mesh(mesh_path)
    mesh = normalise_mesh_step2(mesh)
    mesh_feats = compute_all_features_one_shape(mesh, mesh_name)
    normed_feats = normalise_feat(mesh_feats)
    mesh_distances = compute_single_distance(normed_feats, normed_feats_path="./normalised_features.csv")
    del mesh_distances['file_name']
    del mesh_distances['shape_number']

    closest_meshes = []
    sorted_mesh_dist = {key:val for key, val in sorted(mesh_distances.items(), key=lambda item: item[1])}
    for i in range(num_closest_meshes):
        closest_meshes.append((list(sorted_mesh_dist)[i+1][8:], list(sorted_mesh_dist.values())[i+1]))

    return closest_meshes, mesh


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
        x_transl = (i%5+1)*1.1
        z_transl = -int(i/5)
        cmesh = cmesh.translate(translation=[(i%5+1)*1.1, 0, -int(i/5)])
        cmesh.compute_vertex_normals()
        mesh_objs.append(cmesh)

    o3d.visualization.draw_geometries(mesh_objs)



def query_interface():
    print("Querying meshes")
    print("1) Query a mesh from the database.")
    print("2) Load your own mesh to query.")

    number_of_choices = 2
    possible_choices = [i + 1 for i in range(number_of_choices)]
    choice1 = int(input("\nChoice: "))

    while choice1 not in possible_choices:
        print("\nError! Invalid choice")
        choice1 = int(input("\nChoice: "))

    if choice1 == 1:
        print("lol")

    elif choice1 == 2:
        raise Exception("Not implemented")





#normalise_all_feats("./all_features_small.csv", save_feats=True)
#compute_all_distances("./normalised_features.csv", save=True)
#mesh = load_mesh("./benchmark/db_ref_normalised/0/m0/m0.off")

closest_meshes, mesh_name = query_db_mesh(mesh_name="m0", distance_db_path="distance_to_meshes.csv")
display_query(closest_meshes, mesh_name)