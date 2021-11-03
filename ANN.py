from annoy import AnnoyIndex
from Mesh_Reading import load_mesh
import pandas as pd
from Compute_Features import compute_all_features_one_shape
from Normalization import normalise_mesh_step2
from Query_Meshes import normalise_feat


def ann(query_mesh, feature_list, num_of_trees=100, top_k=10, search_k=100, query=False, metric='euclidean'):
    """
    query_mesh can either be a path to a new shape or a name (like m271, etc.) of a shape we already know.
    whether to extract the features or retrieve them from the feature_list is known by the parameter query:
    if True, the shape is new, if False, the shape is present in feature_list.
    """

    all_feat = pd.read_csv(feature_list, header=0)

    if query == False:

        non_features = ['file_name', 'shape_number', 'Class']

        mesh_feat = []

        df_mesh = all_feat.loc[(all_feat['file_name'] == query_mesh)]

        for feature in df_mesh:
            if feature not in non_features and 'range' not in feature:
                mesh_feat.append(df_mesh[feature].item())

    else:
        mesh = load_mesh(query_mesh)
        mesh = normalise_mesh_step2(mesh)
        mesh_feats = compute_all_features_one_shape(mesh, "query_mesh")
        normed_feats = normalise_feat(mesh_feats)
        mesh_feat = []
        # to be finished... make sure the features are in a list, not a dictionary. Also, normalise the features.


    num_of_features = len(mesh_feat)

    a = AnnoyIndex(num_of_features, metric)
    a.add_item(0, mesh_feat)

    for index, shape in all_feat.iterrows():

        feature_vect = [shape[feature] for feature in all_feat.columns if feature not in non_features and 'range' not in feature]
        a.add_item(shape["shape_number"], feature_vect)

    a.build(num_of_trees)

    results = a.get_nns_by_item(0, top_k + 1, search_k=search_k, include_distances=True)

    similar_shapes = {}

    for shape_number, distance in list(zip(results[0], results[1])):
        similar_shapes['m' + str(shape_number)] = distance

    return similar_shapes


feature_list = "./all_features.csv"
shapes = ann("m99", feature_list)
print(shapes)
