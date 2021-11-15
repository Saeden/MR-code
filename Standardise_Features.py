import csv
import pandas as pd
import numpy as np
import os


def normalise_all_feats(feat_path, bin_number=15, save_feats=False, save_params=True):

    feats = pd.read_csv(feat_path, header=0)
    norm_params = {}
    all_feats = []
    hist_feats = ['a3_', 'd1_', 'd2_', 'd3_', 'd4_']
    hist_dist = {'a3_': [], 'd1_': [], 'd2_': [], 'd3_': [], 'd4_': []}

    weights = get_weights()
    #weights = [0.2, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1, 0.8, 0.01, 0.05, 0.5, 0.4]
    #weights = [0.125, 0.0125, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2, 0.0125, 0.05, 0.1, 0.2]
    #weights = [1,1,1,1,1,1,1,1,1,1,1,1,1]


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
    norm_params['avg_compactness'] = avg_compactness
    norm_params['std_compactness'] = std_compactness

    avg_sphericity = sum(feats['sphericity'])/len(feats['sphericity'])
    std_sphericity = np.std(feats['sphericity'])
    norm_params['avg_sphericity'] = avg_sphericity
    norm_params['std_sphericity'] = std_sphericity

    avg_diameter = sum(feats['diameter'])/len(feats['diameter'])
    std_diameter = np.std(feats['diameter'])
    norm_params['avg_diameter'] = avg_diameter
    norm_params['std_diameter'] = std_diameter

    avg_aabb_vol = sum(feats['aabbox_volume'])/len(feats['aabbox_volume'])
    std_aabb_vol = np.std(feats['aabbox_volume'])
    norm_params['avg_aabb_vol'] = avg_aabb_vol
    norm_params['std_aabb_vol'] = std_aabb_vol

    avg_rect = sum(feats['rectangularity'])/len(feats['rectangularity'])
    std_rect = np.std(feats['rectangularity'])
    norm_params['avg_rect'] = avg_rect
    norm_params['std_rect'] = std_rect

    avg_eccent = sum(feats['eccentricity'])/len(feats['eccentricity'])
    std_eccent = np.std(feats['eccentricity'])
    norm_params['avg_eccent'] = avg_eccent
    norm_params['std_eccent'] = std_eccent


    """print("Finding normalisation parameters.")
    printed = False
    for index1, row1 in feats.iterrows():
        completion = int(((index1+1)/len(feats.index))*100)
        if completion%5 == 0 and not printed:
            print(f"Found {completion}% of normalisation parameters.")
            printed = True
        if completion%5 == 1:
            printed = False
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
    norm_params['std_d4'] = np.std(hist_dist['d4_'])"""

    printed = False
    print("\nStarting normalisation")
    for index, row in feats.iterrows():
        completion = int(((index+1) / len(feats.index)) * 100)
        if completion % 5 == 0 and not printed:
            print(f"Normalised {completion}% of features.")
            printed = True
        if completion % 5 == 1:
            printed = False
        norm_feats = {}
        norm_feats['file_name'] = row['file_name']
        norm_feats['shape_number'] = row['shape_number']
        norm_feats['area'] = ((row['area']-avg_area)/std_area) * weights[0]
        norm_feats['volume'] = ((row['volume']-avg_volume)/std_volume) * weights[1]
        norm_feats['compactness'] = ((row['compactness']-avg_compactness)/std_compactness) * weights[2]
        norm_feats['sphericity'] = ((row['sphericity']-avg_sphericity)/std_sphericity) * weights[3]
        norm_feats['diameter'] = ((row['diameter']-avg_diameter)/std_diameter) * weights[4]
        norm_feats['aabbox_volume'] = ((row['aabbox_volume']-avg_aabb_vol)/std_aabb_vol) * weights[5]
        norm_feats['rectangularity'] = ((row['rectangularity']-avg_rect)/std_rect) * weights[6]
        norm_feats['eccentricity'] = ((row['eccentricity']-avg_eccent)/std_eccent) * weights[7]


        for i in range(bin_number):
            norm_feats[f"a3_{i + 1}"] = row[f"a3_{i + 1}"] * weights[8]
            norm_feats[f"d1_{i + 1}"] = row[f"d1_{i + 1}"] * weights[9]
            norm_feats[f"d2_{i + 1}"] = row[f"d2_{i + 1}"] * weights[10]
            norm_feats[f"d3_{i + 1}"] = row[f"d3_{i + 1}"] * weights[11]
            norm_feats[f"d4_{i + 1}"] = row[f"d4_{i + 1}"] * weights[12]

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

    if not os.path.isfile(norm_param_path):
        print("\nNormalisation parameters have not been calculated.")
        _, norm_params = normalise_all_feats("./all_features.csv", save_feats=True)

    else:
        with open(norm_param_path) as f:
            norm_params = [{k: v for k, v in row.items()}
                           for row in csv.DictReader(f)][0]

    weights = get_weights()
    norm_feats = {}
    norm_feats['file_name'] = feats['file_name']
    norm_feats['shape_number'] = feats['shape_number']
    norm_feats['area'] = (feats['area']-float(norm_params['avg_area']))/float(norm_params['std_area']) * weights[0]
    norm_feats['volume'] = (feats['volume']-float(norm_params['avg_volume']))/float(norm_params['std_volume']) * weights[1]
    norm_feats['compactness'] = (feats['compactness']-float(norm_params['avg_compactness']))/float(norm_params['std_compactness']) * weights[2]
    norm_feats['sphericity'] = (feats['sphericity']-float(norm_params['avg_sphericity']))/float(norm_params['std_sphericity']) * weights[3]
    norm_feats['diameter'] = (feats['diameter']-float(norm_params['avg_diameter']))/float(norm_params['std_diameter']) * weights[4]
    norm_feats['aabbox_volume'] = (feats['aabbox_volume']-float(norm_params['avg_aabb_vol']))/float(norm_params['std_aabb_vol']) * weights[5]
    norm_feats['rectangularity'] = (feats['rectangularity']-float(norm_params['avg_rect']))/float(norm_params['std_rect']) * weights[6]
    norm_feats['eccentricity'] = (feats['eccentricity']-float(norm_params['avg_eccent']))/float(norm_params['std_eccent']) * weights[7]

    for i in range(bin_number):
        norm_feats[f"a3_{i+1}"] = feats[f"a3_{i+1}"] * weights[8]
        norm_feats[f"d1_{i + 1}"] = feats[f"d1_{i + 1}"] * weights[9]
        norm_feats[f"d2_{i + 1}"] = feats[f"d2_{i + 1}"] * weights[10]
        norm_feats[f"d3_{i + 1}"] = feats[f"d3_{i + 1}"] * weights[11]
        norm_feats[f"d4_{i + 1}"] = feats[f"d4_{i + 1}"] * weights[12]

    print("Finished normalising features.")

    return norm_feats


def get_weights():

    return [0.1, 0.025, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2, 0.025, 0.05, 0.1, 0.2]

#normalise_all_feats(feat_path='./all_features.csv', save_feats=True)