from sklearn import manifold
import pandas as pd
from utils import get_complete_classification
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc
import mplcursors


def dimensionality_reduction(feature_list):

    print("Computing the scatterplot...")

    all_feat = pd.read_csv(feature_list, header=0)
    non_features = ['file_name', 'shape_number']

    labels_dictionary = get_complete_classification()
    classes = []
    mesh_names = []
    unique_classes = []

    # get the mesh name and class for each mesh in the database
    for mesh_name in all_feat['file_name'].values:
        mesh_names.append(mesh_name)
        classes.append(labels_dictionary[int(mesh_name[1:])])

    # get the unique classes
    for class_ in classes:
        if class_ not in unique_classes:
            unique_classes.append(class_)

    # drop the columns of the non features
    all_feat = all_feat.drop(columns=non_features)
    feature_list = all_feat.values.tolist()

    # perform the tsne dimensionality reduction
    tsne = manifold.TSNE(init="pca", perplexity=19, learning_rate=200)
    features_2D = tsne.fit_transform(feature_list)

    x = features_2D[:, 0]
    y = features_2D[:, 1]

    df = pd.DataFrame()

    df["x"] = x
    df["y"] = y

    df["label"] = classes

    # plot the results
    palette = sns.color_palette(cc.glasbey, n_colors=54)

    graph = sns.scatterplot(x="x", y="y", hue=classes, palette=palette, data=df)
    graph.legend(fontsize='6', loc='right', bbox_to_anchor=(1.13, 0.5), ncol=1)

    plt.title("Dimensionality Reduction for 54 classes", fontdict={'fontsize': 20})

    print("Scatterplot computed.")

    # view the mesh name and class of a selected point on the scatterplot (on hover)
    mplcursors.cursor(graph, hover=True).connect("add", lambda sel: cursor_annotations(sel, mesh_names, classes))

    plt.show()


def cursor_annotations(sel, mesh_names, classes):

    sel.annotation.set_text(f"Mesh: {mesh_names[sel.index]} | Class: {classes[sel.index]}")
    sel.annotation.get_bbox_patch().set(fc="powderblue", alpha=0.9)


dimensionality_reduction("./normalised_features.csv")