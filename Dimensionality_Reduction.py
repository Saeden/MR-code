from sklearn import manifold
import pandas as pd
from utils import get_complete_classification
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc


def dimensionality_reduction(feature_list):

    all_feat = pd.read_csv(feature_list, header=0)

    non_features = ['file_name', 'shape_number']

    labels_dictionary = get_complete_classification()

    classes = []

    for mesh_name in all_feat['file_name'].values:
        classes.append(labels_dictionary[int(mesh_name[1:])])

    print(classes)

    all_feat = all_feat.drop(columns=non_features)

    feature_list = all_feat.values.tolist()

    tsne = manifold.TSNE(init="pca", perplexity=19, learning_rate=200)
    features_2D = tsne.fit_transform(feature_list)

    x = features_2D[:, 0]
    y = features_2D[:, 1]

    df = pd.DataFrame()

    df["x"] = x
    df["y"] = y

    df["label"] = classes

    palette = sns.color_palette(cc.glasbey, n_colors=54)

    graph = sns.scatterplot(x="x", y="y", hue=classes, palette=palette, data=df)
    graph.legend(fontsize='6', loc='right', bbox_to_anchor=(1.13, 0.5), ncol=1)

    plt.title("Dimensionality Reduction for 54 classes", fontdict={'fontsize' : 20})
    plt.show()
