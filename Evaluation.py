from Query_Meshes import query_db_mesh
import pandas as pd
import time


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


def evaluate_db(mesh_db_path="./all_features.csv"):
    time1 = time.time()
    mesh_db = pd.read_csv(mesh_db_path, header=0)
    db_size = len(mesh_db['file_name'])
    class_count_dict, class_metrics = class_count(mesh_db['Class'].tolist())
    #truth_table = {'True_Pos':0, 'True_Neg': 0, 'False_Pos':0, 'False_Neg':0}

    avg_metrics = {'avg_accuracy': 0, 'avg_sensitivity': 0, 'avg_specificity': 0}
    class_acc = class_metrics.copy()
    class_sens = class_metrics.copy()
    class_spec = class_metrics.copy()
    del class_metrics

    index = 0
    printed = False

    for mesh_name in mesh_db['file_name']:
        truth_table = {'True_Pos': 0, 'True_Neg': 0, 'False_Pos': 0, 'False_Neg': 0}
        index += 1
        completion = int(((index) / db_size * 100))
        if completion % 5 == 0 and not printed:
            print(f"Evaluated {completion}% of meshes.")
            printed = True
        if completion % 5 == 1:
            printed = False

        mesh_class = mesh_db['Class'].loc[mesh_db['file_name'] == mesh_name].item()
        closest_meshes, mesh_name = query_db_mesh(mesh_name=mesh_name, num_closest_meshes=25)

        closest_mesh_classes = [mesh_db['Class'].loc[mesh_db['file_name'] == closest_meshes[i][0]].item()
                                for i in range(len(closest_meshes))]
        for item in closest_mesh_classes:
            if mesh_class == item:
                truth_table['True_Pos'] += 1
            else:
                truth_table['False_Pos'] += 1

        truth_table['False_Neg'] = class_count_dict[mesh_class] - truth_table['True_Pos']
        truth_table['True_Neg'] = db_size - (truth_table['True_Pos'] + truth_table['False_Neg'] + truth_table['False_Pos'])

        accuracy = (truth_table['True_Pos'])/class_count_dict[mesh_class]
        #(truth_table['True_Pos']+truth_table['True_Neg'])/db_size
        class_acc[mesh_class] += accuracy
        avg_metrics['avg_accuracy'] += accuracy

        sensitivity = truth_table['True_Pos']/(truth_table['True_Pos'] + truth_table['False_Neg'])
        avg_metrics['avg_sensitivity'] += sensitivity
        class_sens[mesh_class] += sensitivity

        specificity = truth_table['True_Neg']/(truth_table['False_Pos'] + truth_table['True_Neg'])
        avg_metrics['avg_specificity'] += specificity
        class_spec[mesh_class] += specificity

    for item in list(class_count_dict):
        class_acc[item] /= class_count_dict[item]
        class_sens[item] /= class_count_dict[item]
        class_spec[item] /= class_count_dict[item]

    for key in list(avg_metrics):
        avg_metrics[key] /= db_size

    time2 = time.time()
    print(f"Time to evaluate: {time2-time1}")


    print("\nAverage metrics")
    print(avg_metrics)

    print("\nClass accuracy:")
    print(class_acc)
    print("\nClass sensitivity:")
    print(class_sens)
    print("\nClass specificity:")
    print(class_spec)













evaluate_db()

#[(closest_meshes[i][0], mesh_db['Class'].loc[mesh_db['file_name'] == closest_meshes[i][0]].item()) for i in range(len(closest_meshes))]