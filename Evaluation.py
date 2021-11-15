from Query_Meshes import query_db_mesh_fast
from ANN import ann_fast
from sklearn.metrics import auc
import pandas as pd
import time
import matplotlib.pyplot as plt


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


def evaluate_db(mesh_db_path="./all_features.csv", query_num=25):
    time1 = time.time()
    mesh_db = pd.read_csv(mesh_db_path, header=0)
    db_size = len(mesh_db['file_name'])
    class_count_dict, class_metrics = class_count(mesh_db['Class'].tolist())

    avg_metrics = {'avg_accuracy': 0, 'avg_sensitivity': 0, 'avg_specificity': 0, 'avg_precision':0, 'avg_f1':0}
    class_acc = class_metrics.copy()
    class_sens = class_metrics.copy()
    class_spec = class_metrics.copy()
    class_prec = class_metrics.copy()
    class_f1sc = class_metrics.copy()
    del class_metrics

    distance_db = pd.read_csv("./distance_to_meshes.csv", header=0)

    index = 0
    printed = False

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


    print("\nAverage metrics")
    print(avg_metrics)

    """print("\nClass accuracy:")
    print(class_acc)
    print("\nClass sensitivity:")
    print(class_sens)
    print("\nClass specificity:")
    print(class_spec)
    print("\nClass precision")
    print(class_prec)
    print("\nClass F1-score")
    print(class_f1sc)"""

    results = [avg_metrics, class_acc, class_sens, class_spec, class_prec, class_f1sc, "DIST_RESULTS"]

    return results

def evaluate_ann(mesh_db_path="./all_features.csv", query_num=25):
    time1 = time.time()
    mesh_db = pd.read_csv(mesh_db_path, header=0)
    db_size = len(mesh_db['file_name'])
    class_count_dict, class_metrics = class_count(mesh_db['Class'].tolist())

    avg_metrics = {'avg_accuracy': 0, 'avg_sensitivity': 0, 'avg_specificity': 0, 'avg_precision': 0, 'avg_f1':0}
    class_acc = class_metrics.copy()
    class_sens = class_metrics.copy()
    class_spec = class_metrics.copy()
    class_prec = class_metrics.copy()
    class_f1sc = class_metrics.copy()
    del class_metrics

    map = pd.read_csv("mapping.csv", header=0)
    features = pd.read_csv("normalised_features.csv", header=0)

    index = 0
    printed = False

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

    print("\nAverage metrics")
    print(avg_metrics)

    """print("\nClass accuracy:")
    print(class_acc)
    print("\nClass sensitivity:")
    print(class_sens)
    print("\nClass specificity:")
    print(class_spec)
    print("\nClass precision")
    print(class_prec)
    print("\nClass F1-score")
    print(class_f1sc)"""

    results = [avg_metrics, class_acc, class_sens, class_spec, class_prec, class_f1sc, "ANN_RESULTS"]

    return results


def compute_roc_curve():
    query_nums = [5, 10, 25, 50, 75, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    tpr_dist = [0]
    fpr_dist = [0]
    tpr_ann = [0]
    fpr_ann = [0]
    all_dist_results = []
    all_ann_results = []
    for num in query_nums:
        result_dist = evaluate_db(query_num=num)
        all_dist_results.append((len(all_dist_results), result_dist))
        tpr_dist.append(result_dist[0]['avg_sensitivity'])
        fpr_dist.append(1-result_dist[0]['avg_specificity'])


        result_ann = evaluate_ann(query_num=num)
        all_ann_results.append((len(all_ann_results), result_ann))
        tpr_ann.append(result_ann[0]['avg_sensitivity'])
        fpr_ann.append(1-result_ann[0]['avg_specificity'])

    tpr_dist.append(1)
    fpr_dist.append(1)
    tpr_ann.append(1)
    fpr_ann.append(1)


    auc_dist = auc(fpr_dist, tpr_dist)
    auc_ann = auc(fpr_ann, tpr_ann)

    plt.plot(fpr_dist, tpr_dist, label="AUC_que=" + str(auc_dist))
    plt.plot(fpr_ann, tpr_ann,  label="AUC_ann=" + str(auc_ann))
    plt.ylabel('Sensitivity')
    plt.xlabel('1-Specificity')
    plt.legend(loc=4)
    plt.show()














compute_roc_curve()
#evaluate_db()
#evaluate_ann()
#[(closest_meshes[i][0], mesh_db['Class'].loc[mesh_db['file_name'] == closest_meshes[i][0]].item()) for i in range(len(closest_meshes))]