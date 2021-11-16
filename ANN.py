from annoy import AnnoyIndex
from Mesh_Reading import *
import pandas as pd
from Compute_Features import compute_all_features_one_shape
from Normalization import normalise_mesh_step2
from Standardise_Features import normalise_feat
from Mesh_refining import refine_single_mesh
from utils import *
import csv

"""
There are just two main parameters needed to tune Annoy: the number of trees n_trees and 
the number of nodes to inspect during searching search_k.

n_trees is provided during build time and affects the build time and the index size. A larger value will give more 
accurate results, but larger indexes. search_k is provided in runtime and affects the search performance. 
A larger value will give more accurate results, but will take longer time to return. If search_k is not provided, 
it will default to k * n_trees where k is the number of approximate nearest neighbors. 
Otherwise, search_k and n_trees are roughly independent, i.e. the value of n_trees will not affect search time if 
search_k is held constant and vice versa. Basically it's recommended to set n_trees as large as possible given 
the amount of memory you can afford, and it's recommended to set search_k as large as possible given the time 
constraints you have for the queries.

search_k=-1 means default.
"""

def ann(query_mesh, feature_list, num_of_trees=1000, top_k=25, search_k=-1, query=False, metric='euclidean'):
    """
    query_mesh can either be a path to a new shape, or a name (like m271, etc.) of a shape we already know.
    whether to extract the features or retrieve them from the feature_list (ideally, the normalized version)
    is known by the parameter query: if True, the shape is new, if False, the shape is present in feature_list.
    """


    all_feat = pd.read_csv(feature_list, header=0)

    non_features = ['file_name', 'shape_number', 'Class']

    if query == False:
        # then it's a shape we have in the feature_list DB, simply retrieve its features
        mesh_feat = []

        df_mesh = all_feat.loc[(all_feat['file_name'] == query_mesh)]
        shape_path = get_path_from_shape(query_mesh, "db_ref_normalised")
        norm_mesh = load_mesh(shape_path)

        for feature in df_mesh:
            if feature not in non_features and 'range' not in feature:
                mesh_feat.append(df_mesh[feature].item())

    else:
        # it's a new shape, we have to re-mesh it, normalize it, extract the features, normalized them and store them in a list
        mesh = load_mesh(query_mesh)
        print("\nRe-meshing the shape.")
        ref_mesh = refine_single_mesh(mesh)
        print("\nNormalising mesh.")
        norm_mesh = normalise_mesh_step2(ref_mesh)
        print("\nComputing features for new shape.")
        mesh_feats = compute_all_features_one_shape(norm_mesh, "m0000")
        print("\nNormalising features of new shape.")
        normed_feats = normalise_feat(mesh_feats)
        mesh_feat_raw = list(normed_feats.values())

        mesh_feat = mesh_feat_raw[2:]  # remove the first 2 elements, namely, file_name and shape_number, from the list of features

    num_of_features = len(mesh_feat)

    # map a shape with a fictionary index: eg, 0: m99, 1: m100 etc. If the shape is new, the mapping
    # for that shape will have the index 0 and the name will be the path of that new shape.
    mapping = {}

    # if we don't already have built a forest, let's build one
    if not os.path.exists('./query_forest_' + metric + '_' + str(num_of_trees) + '.ann'):

        # create the annoy with a shape of the len of the features vector and the specified distance metric
        # metric can be "angular", "euclidean", "manhattan", "hamming", or "dot"
        a = AnnoyIndex(num_of_features, metric)

        # add the shape we want to query to the annoy function with the index 0 and its feature vector
        a.add_item(0, mesh_feat)
        i = 1
        for index, shape in all_feat.iterrows():
            # add all the shapes in the feature_list with the fictionary index and their features vectors
            # the mapping between index and name of the shape is provided by the dictionary 'mapping'
            mapping[i] = shape['file_name']
            feature_vect = [shape[feature] for feature in all_feat.columns if feature not in non_features and 'range' not in feature]
            a.add_item(i, feature_vect)
            i += 1

        # build the ann forest with the specified number of trees and save it to then open it later when needed
        # by saving it, we will be faster when query because we don't have to construct again the whole forest
        a.build(num_of_trees)
        a.save('query_forest_' + metric + '_' + str(num_of_trees) + '.ann')

        # we also need to store the mapping in order to have the same mapping when loading the saved forest
        with open("mapping.csv", 'w', encoding='UTF8', newline='') as f:
            output = [mapping]
            fieldnames = [i for i in mapping]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output)

        # now we can retrieve the first k results with minimum distance compared to the query shape.
        # during the query time, it will inspect up to search_k nodes which defaults to n_trees * k if not provided.
        # search_k gives you a run-time trade-off between better accuracy and speed.
        # the query is performed by index: find the k shapes that are closest to the shape with index 0, the query shape
        mapping[0] = query_mesh
        results = a.get_nns_by_item(0, top_k + 2, search_k=search_k, include_distances=True)

    # if we already have a forest in our system, load it so that we don't have to build a new one
    else:
        # create the annoy with a shape of the len of the features vector and the specified distance metric
        # metric can be "angular", "euclidean", "manhattan", "hamming", or "dot"
        # the metric should be the same as the one used in the saved file.
        a = AnnoyIndex(num_of_features, metric)
        a.load('query_forest_' + metric + '_' + str(num_of_trees) + '.ann')
        # load also the mapping and transform it in a dictionary, then assign the index 0 to the shape we want to query
        map = pd.read_csv("mapping.csv", header=0)
        for i in map:
            mapping[int(i)] = map[i].item()
        mapping[0] = query_mesh

        # now we perform the query with the vector of extracted features of the new shape in input
        results = a.get_nns_by_vector(mesh_feat, top_k + 2, search_k=search_k, include_distances=True)

    # now the query is done and we can retrieve the similar shapes by unpacking the result of ANN.
    # with a dictionary, we can get rid of all the duplicates, if there are any
    similar_shapes_raw = {}

    for i, distance in list(zip(results[0], results[1])):
        similar_shapes_raw[mapping[i]] = distance

    # transform the dictionary in a tuple so to fit the display_query requirements
    similar_shapes = [(key, value) for (key, value) in similar_shapes_raw.items()]

    return similar_shapes, norm_mesh


def ann_fast(query_mesh, features, map, num_of_trees=1000, top_k=10, search_k=-1, metric='euclidean'):

    all_feat = features
    non_features = ['file_name', 'shape_number', 'Class']

    mesh_feat = []

    df_mesh = all_feat.loc[(all_feat['file_name'] == query_mesh)]

    for feature in df_mesh:
        if feature not in non_features and 'range' not in feature:
            mesh_feat.append(df_mesh[feature].item())

    num_of_features = len(mesh_feat)

    mapping = {}

    a = AnnoyIndex(num_of_features, metric)
    a.load('query_forest_' + metric + '_' + str(num_of_trees) + '.ann')

    for i in map:
        mapping[int(i)] = map[i].item()
    mapping[0] = query_mesh

    results = a.get_nns_by_vector(mesh_feat, top_k + 2, search_k=search_k, include_distances=True)

    similar_shapes_raw = {}

    for i, distance in list(zip(results[0], results[1])):
        similar_shapes_raw[mapping[i]] = distance

    similar_shapes = [(key, value) for (key, value) in similar_shapes_raw.items()]

    return similar_shapes
