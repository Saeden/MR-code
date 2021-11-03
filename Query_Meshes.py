import pandas as pd
import csv
import numpy as np


def normalise_all_feats(feat_path, bin_number=15, save_feats=False, save_params=True):
    feats = pd.read_csv(feat_path, header=0)
    norm_params = {}
    all_feats = []

    avg_area = sum(feats['area'])/len(feats['area'])
    std_area = np.std(feats['area'])
    norm_params['avg_area'] = avg_area
    norm_params['std_area'] = std_area

    avg_volume = sum(feats['volume'])/len(feats['volume'])
    std_volume = np.std(feats['volume'])
    norm_params['avg_volume'] =avg_volume
    norm_params['std_volume'] =std_volume

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

    for index, row in feats.iterrows():
        norm_feats = {}
        norm_feats['file_name'] = row['file_name']
        norm_feats['shape_number'] = row['shape_number']
        norm_feats['area'] = (row['area']-std_area)/avg_area
        norm_feats['volume'] = (row['volume']-std_volume)/avg_volume
        norm_feats['compactness'] = (row['compactness']-std_compactness)/avg_compactness
        norm_feats['sphericity'] = (row['sphericity']-std_sphericity)/avg_sphericity
        norm_feats['diameter'] = (row['diameter']-std_diameter)/avg_diameter
        norm_feats['aabbox_volume'] = (row['aabbox_volume']-std_aabb_vol)/avg_aabb_vol
        norm_feats['rectangularity'] = (row['rectangularity']-std_rect)/avg_rect
        norm_feats['eccentricity'] = (row['eccentricity']-std_eccent)/avg_eccent

        total_a3 = 0
        total_d1 = 0
        total_d2 = 0
        total_d3 = 0
        total_d4 = 0

        for b in range(bin_number):
            total_a3 += row[f"a3_{b+1}"]
            total_d1 += row[f"d1_{b+1}"]
            total_d2 += row[f"d2_{b+1}"]
            total_d3 += row[f"d3_{b+1}"]
            total_d4 += row[f"d4_{b+1}"]

        for i in range(bin_number):
            #divide by bin number to make the weight of each histogram equal the weight of each scalar feature
            norm_feats[f"a3_{i+1}"] = row[f"a3_{i+1}"]/total_a3/bin_number
            norm_feats[f"d1_{i + 1}"] = row[f"d1_{i + 1}"]/total_d1/bin_number
            norm_feats[f"d2_{i + 1}"] = row[f"d2_{i + 1}"]/total_d2/bin_number
            norm_feats[f"d3_{i + 1}"] = row[f"d3_{i + 1}"]/total_d3/bin_number
            norm_feats[f"d4_{i + 1}"] = row[f"d4_{i + 1}"]/total_d4/bin_number

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

    return all_feats


def compute_all_distances(normed_feats_path=None, mesh=None, bin_number=15):
    global_feats = ['area', 'volume', 'compactness', 'sphericity', 'diameter', 'aabbox_volume', 'rectangularity',
                    'eccentricity']
    hist_feats = ['a3_', 'd1_', 'd2_', 'd3_', 'd4_']
    if normed_feats_path == None:
        normed_feats = normalise_all_feats("./all_features.csv")

    else:
        with open(normed_feats_path) as f:
            normed_feats = [{k: v for k, v in row.items()}
                 for row in csv.DictReader(f)]

    for feat1 in normed_feats:
        dist_to_meshes = {'file_name': feat1['file_name'], 'shape_number': feat1['shape_number']}
        for feat2 in normed_feats:
            global_dist = 0
            hist_dist = []
            if feat1['file_name'] == feat2['file_name']:
                continue
            else:
                for gf in global_feats:
                    global_dist += (float(feat1[gf])-float(feat2[gf]))**2
                global_dist = np.sqrt(global_dist)

                for hf in hist_feats:
                    temp_dist = 0
                    for i in range(bin_number):
                        temp_dist += (float(feat1[str(hf+str(i+1))])-float(feat2[str(hf+str(i+1))]))**2
                    hist_dist.append(np.sqrt(temp_dist))

                total_dist = (sum(hist_dist)+global_dist)/len(hist_dist)+1

                dist_to_meshes[f"dist_to{feat2['file_name']}"] = total_dist











#normalise_all_feats("./all_features.csv", save_feats=True)
#compute_all_distances("./normalised_features.csv")