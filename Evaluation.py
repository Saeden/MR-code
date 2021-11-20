from Query_Meshes import query_db_mesh_fast
from ANN import ann_fast
from sklearn.metrics import auc
import pandas as pd
import time
import matplotlib.pyplot as plt
from copy import deepcopy as dcopy


def class_count(class_list):
    count_dict = {}
    class_metrics = {}
    for item in class_list:
        if item not in list(count_dict):
            count_dict[item] = 1
            class_metrics[item] = 0
        else:
            count_dict[item] += 1

    return count_dict, class_metrics


def evaluate_db(mesh_db_path="./all_features.csv", query_num=25, print_all=True):
    time1 = time.time()
    mesh_db = pd.read_csv(mesh_db_path, header=0)
    db_size = len(mesh_db['file_name'])
    class_count_dict, class_metrics = class_count(mesh_db['Class'].tolist())

    avg_metrics = {'avg_accuracy': 0, 'avg_sensitivity': 0, 'avg_specificity': 0, 'avg_precision':0, 'avg_f1':0}
    class_acc = dcopy(class_metrics)
    class_sens = dcopy(class_metrics)
    class_spec = dcopy(class_metrics)
    class_prec = dcopy(class_metrics)
    class_f1sc = dcopy(class_metrics)
    del class_metrics

    distance_db = pd.read_csv("./distance_to_meshes.csv", header=0)

    index = 0
    printed = False
    print('\n')
    for mesh_name in mesh_db['file_name']:
        #truth_table = {'True_Pos': 0, 'True_Neg': 0, 'False_Pos': 0, 'False_Neg': 0}
        truth_table = [0, 0, 0, 0]  #[TP, TN, FP, FN]
        index += 1
        completion = int((index / db_size * 100))
        if completion % 5 == 0 and not printed:
            print(f"Evaluated {completion}% of meshes.")
            printed = True
        if completion % 5 == 1:
            printed = False

        mesh_class = mesh_db['Class'].loc[mesh_db['file_name'] == mesh_name].item()
        closest_meshes, mesh_name = query_db_mesh_fast(mesh_name=mesh_name, distance_db=distance_db, num_closest_meshes=query_num)

        closest_mesh_classes = [mesh_db['Class'].loc[mesh_db['file_name'] == closest_meshes[i][0]].item()
                                for i in range(len(closest_meshes))]
        for item in closest_mesh_classes:
            if mesh_class == item:
                truth_table[0] += 1
            else:
                truth_table[2] += 1

        truth_table[3] = class_count_dict[mesh_class] - truth_table[0]
        truth_table[1] = db_size - (truth_table[0] + truth_table[3] + truth_table[2])

        accuracy = (truth_table[0]+truth_table[1])/db_size
        class_acc[mesh_class] += accuracy
        avg_metrics['avg_accuracy'] += accuracy

        sensitivity = truth_table[0]/(truth_table[0] + truth_table[3])
        avg_metrics['avg_sensitivity'] += sensitivity
        class_sens[mesh_class] += sensitivity

        specificity = truth_table[1]/(truth_table[2] + truth_table[1])
        avg_metrics['avg_specificity'] += specificity
        class_spec[mesh_class] += specificity

        precision = truth_table[0]/(truth_table[0]+truth_table[2])
        avg_metrics['avg_precision'] += precision
        class_prec[mesh_class] += precision

        if precision + sensitivity == 0:
            f1_score = 0
        else:
            f1_score = 2 * ((precision * sensitivity) / (precision + sensitivity))
        avg_metrics['avg_f1'] += f1_score
        class_f1sc[mesh_class] += f1_score

    for item in list(class_count_dict):
        class_acc[item] /= class_count_dict[item]
        class_sens[item] /= class_count_dict[item]
        class_spec[item] /= class_count_dict[item]
        class_prec[item] /= class_count_dict[item]
        class_f1sc[item] /= class_count_dict[item]

    for key in list(avg_metrics):
        avg_metrics[key] /= db_size

    time2 = time.time()
    print(f"Time to evaluate: {time2-time1}")


    print(f"\nAverage metrics DIST for q-size: {query_num}")
    print(avg_metrics)


    if print_all:
        print("\nClass accuracy:")
        print(class_acc)
        print("\nClass sensitivity:")
        print(class_sens)
        print("\nClass specificity:")
        print(class_spec)
        print("\nClass precision")
        print(class_prec)
        print("\nClass F1-score")
        print(class_f1sc)

    results = [avg_metrics, class_acc, class_sens, class_spec, class_prec, class_f1sc, "DIST_RESULTS"]

    return results


def evaluate_ann(mesh_db_path="./all_features.csv", query_num=25, print_all=True):

    time1 = time.time()
    mesh_db = pd.read_csv(mesh_db_path, header=0)
    db_size = len(mesh_db['file_name'])
    class_count_dict, class_metrics = class_count(mesh_db['Class'].tolist())

    avg_metrics = {'avg_accuracy': 0, 'avg_sensitivity': 0, 'avg_specificity': 0, 'avg_precision': 0, 'avg_f1':0}
    class_acc = dcopy(class_metrics)
    class_sens = dcopy(class_metrics)
    class_spec = dcopy(class_metrics)
    class_prec = dcopy(class_metrics)
    class_f1sc = dcopy(class_metrics)
    del class_metrics

    map = pd.read_csv("mapping.csv", header=0)
    features = pd.read_csv("normalised_features.csv", header=0)

    index = 0
    printed = False
    print('\n')
    for mesh_name in mesh_db['file_name']:
        #truth_table = {'True_Pos': 0, 'True_Neg': 0, 'False_Pos': 0, 'False_Neg': 0}
        truth_table = [0, 0, 0, 0]  #[TP, TN, FP, FN]
        index += 1
        completion = int((index / db_size * 100))
        if completion % 5 == 0 and not printed:
            print(f"Evaluated {completion}% of meshes.")
            printed = True
        if completion % 5 == 1:
            printed = False

        mesh_class = mesh_db['Class'].loc[mesh_db['file_name'] == mesh_name].item()
        closest_meshes = ann_fast(query_mesh=mesh_name, features=features, map=map,
                                  num_of_trees=1000, top_k=query_num)

        closest_mesh_classes = [mesh_db['Class'].loc[mesh_db['file_name'] == closest_meshes[i][0]].item()
                                for i in range(len(closest_meshes))]
        for item in closest_mesh_classes:
            if mesh_class == item:
                truth_table[0] += 1
            else:
                truth_table[2] += 1

        truth_table[3] = class_count_dict[mesh_class] - truth_table[0]
        truth_table[1] = db_size - (truth_table[0] + truth_table[3] + truth_table[2])

        accuracy = (truth_table[0]+truth_table[1])/db_size
        class_acc[mesh_class] += accuracy
        avg_metrics['avg_accuracy'] += accuracy

        sensitivity = truth_table[0]/(truth_table[0] + truth_table[3])
        avg_metrics['avg_sensitivity'] += sensitivity
        class_sens[mesh_class] += sensitivity

        specificity = truth_table[1]/(truth_table[2] + truth_table[1])
        avg_metrics['avg_specificity'] += specificity
        class_spec[mesh_class] += specificity

        precision = truth_table[0]/(truth_table[0]+truth_table[2])
        avg_metrics['avg_precision'] += precision
        class_prec[mesh_class] += precision

        if precision + sensitivity == 0:
            f1_score = 0
        else:
            f1_score = 2 * ((precision * sensitivity) / (precision + sensitivity))
        avg_metrics['avg_f1'] += f1_score
        class_f1sc[mesh_class] += f1_score


    for item in list(class_count_dict):
        class_acc[item] /= class_count_dict[item]
        class_sens[item] /= class_count_dict[item]
        class_spec[item] /= class_count_dict[item]
        class_prec[item] /= class_count_dict[item]
        class_f1sc[item] /= class_count_dict[item]

    for key in list(avg_metrics):
        avg_metrics[key] /= db_size

    time2 = time.time()
    print(f"Time to evaluate: {time2-time1}")

    print(f"\nAverage metrics ANN for q-size: {query_num}")
    print(avg_metrics)

    if print_all:
        print("\nClass accuracy:")
        print(class_acc)
        print("\nClass sensitivity:")
        print(class_sens)
        print("\nClass specificity:")
        print(class_spec)
        print("\nClass precision")
        print(class_prec)
        print("\nClass F1-score")
        print(class_f1sc)

    results = [avg_metrics, class_acc, class_sens, class_spec, class_prec, class_f1sc, "ANN_RESULTS"]

    return results


def compute_roc_curve():
    print("Computing the overall ROC curve for ANN and DIST methods and the ROC curve for all classes.")
    print("This function plots all classes of each method in a graph for each method.")
    query_nums = [1, 5, 10, 25, 50, 75, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
                  1100, 1200, 1300, 1400, 1500, 1600, 1700, 1750]

    class_list = pd.read_csv("all_features.csv", header=0)['Class'].tolist()
    _, class_tpr_fpr = class_count(class_list)

    for key in list(class_tpr_fpr):
        class_tpr_fpr[key] = [[], []]

    dist_class_tpr_fpr = dcopy(class_tpr_fpr)
    ann_class_tpr_fpr = dcopy(class_tpr_fpr)


    tpr_dist = []
    fpr_dist = []
    tpr_ann = []
    fpr_ann = []
    all_dist_results = []
    all_ann_results = []

    for num in query_nums:
        result_dist = evaluate_db(query_num=num, print_all=False)
        all_dist_results.append((len(all_dist_results), result_dist))
        tpr_dist.append(result_dist[0]['avg_sensitivity'])
        fpr_dist.append(1-result_dist[0]['avg_specificity'])



        result_ann = evaluate_ann(query_num=num, print_all=False)
        all_ann_results.append((len(all_ann_results), result_ann))
        tpr_ann.append(result_ann[0]['avg_sensitivity'])
        fpr_ann.append(1-result_ann[0]['avg_specificity'])


        for key in list(dist_class_tpr_fpr):
            dist_class_sens = result_dist[2][key]
            dist_class_spec = result_dist[3][key]
            dist_class_tpr_fpr[key][0].append(dist_class_sens)
            dist_class_tpr_fpr[key][1].append(1-dist_class_spec)

            ann_class_sens = result_ann[2][key]
            ann_class_spec = result_ann[3][key]
            ann_class_tpr_fpr[key][0].append(ann_class_sens)
            ann_class_tpr_fpr[key][1].append(1 - ann_class_spec)


    auc_dist = auc(fpr_dist, tpr_dist)
    auc_ann = auc(fpr_ann, tpr_ann)
    plot0 = plt.figure(1)
    plt.plot(fpr_dist, tpr_dist, label="AUC_dist=" + str(auc_dist))
    plt.plot(fpr_ann, tpr_ann,  label="AUC_ann=" + str(auc_ann))
    plt.ylabel('Sensitivity')
    plt.xlabel('1-Specificity')
    plt.legend(loc=4)
    plt.title(label="ROC curve for ANN and DIST")


    class_auc = []
    plot1 = plt.figure(2)
    for key in list(dist_class_tpr_fpr):
        tpr = dist_class_tpr_fpr[key][0]
        fpr = dist_class_tpr_fpr[key][1]
        this_auc = auc(fpr, tpr)
        class_auc.append((key, this_auc))
        plt.plot(fpr, tpr, label=key)

    print(class_auc)

    plt.ylabel('Sensitivity')
    plt.xlabel('1-Specificity')
    plt.title(label="ROC curve for all classes (DIST)")

    class_auc = []
    plot2 = plt.figure(3)
    for key in list(ann_class_tpr_fpr):
        tpr = ann_class_tpr_fpr[key][0]
        fpr = ann_class_tpr_fpr[key][1]
        this_auc = auc(fpr, tpr)
        class_auc.append((key, this_auc))
        plt.plot(fpr, tpr, label=key)

    print(class_auc)
    plt.ylabel('Sensitivity')
    plt.xlabel('1-Specificity')
    plt.title(label="ROC curve for all classes (ANN)")
    plt.show()




def compute_class_roc_curve():
    print("Computing the ROC curves for all classes with the ANN and DIST methods.")
    print("This function makes a graph for each class, plotting the ANN method against the DIST method.")
    query_nums = [1, 5, 10, 25, 50, 75, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
                  1100, 1200, 1300, 1400, 1500, 1600, 1700, 1750]

    class_list = pd.read_csv("all_features.csv", header=0)['Class'].tolist()
    _, class_tpr_fpr = class_count(class_list)

    for key in list(class_tpr_fpr):
        class_tpr_fpr[key] = [[], []]

    dist_class_tpr_fpr = dcopy(class_tpr_fpr)
    ann_class_tpr_fpr = dcopy(class_tpr_fpr)

    all_dist_results = []
    all_ann_results = []

    for num in query_nums:
        result_dist = evaluate_db(query_num=num, print_all=False)
        all_dist_results.append((len(all_dist_results), result_dist))



        result_ann = evaluate_ann(query_num=num, print_all=False)
        all_ann_results.append((len(all_ann_results), result_ann))


        for key in list(dist_class_tpr_fpr):
            dist_class_sens = result_dist[2][key]
            dist_class_spec = result_dist[3][key]
            dist_class_tpr_fpr[key][0].append(dist_class_sens)
            dist_class_tpr_fpr[key][1].append(1-dist_class_spec)

            ann_class_sens = result_ann[2][key]
            ann_class_spec = result_ann[3][key]
            ann_class_tpr_fpr[key][0].append(ann_class_sens)
            ann_class_tpr_fpr[key][1].append(1 - ann_class_spec)

    class_auc = []
    for index, key in enumerate(list(dist_class_tpr_fpr)):
        tpr = dist_class_tpr_fpr[key][0]
        fpr = dist_class_tpr_fpr[key][1]
        this_auc = auc(fpr, tpr)
        class_auc.append((f"{key}_DIST", this_auc))
        plot = plt.figure(index)
        plt.plot(fpr, tpr, label=f"ROC_DIST: {key}")

        tpr_ann = ann_class_tpr_fpr[key][0]
        fpr_ann = ann_class_tpr_fpr[key][1]
        this_auc_ann = auc(fpr_ann, tpr_ann)
        class_auc.append((f"{key}_ANN", this_auc_ann))
        plt.plot(fpr_ann, tpr_ann, label=f"ROC_ANN: {key}")

        plt.ylabel('Sensitivity')
        plt.xlabel('1-Specificity')
        plt.title(label=f"ROC curve class: {key}")

    print(class_auc)
    plt.show()


    print(class_auc)
    plt.ylabel('Sensitivity')
    plt.xlabel('1-Specificity')
    plt.title(label="ROC curve for all classes (ANN)")
    plt.show()









#compute_roc_curve()
#evaluate_db(query_num=25)
#evaluate_ann(query_num=25)
#[(closest_meshes[i][0], mesh_db['Class'].loc[mesh_db['file_name'] == closest_meshes[i][0]].item()) for i in range(len(closest_meshes))]