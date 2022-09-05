from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import random
import os
import pdb


def plot_kmeans(model, vectors, n_clusters):
    
    
    kmean_indicies = model.fit_predict(vectors)
    pca = PCA(n_components=2)
    
    scatter_plot_points = pca.fit_transform(vectors.toarray())
    
    
    x = [point[0] for point in scatter_plot_points]
    y = [point[1] for point in scatter_plot_points]
    
    get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF),range(n)))
    colors = get_colors(n_clusters)
    
    fig, ax = plt.subplots(figsize=(10,10))
    
    ax.scatter(x, y, c = [colors[d] for d in kmean_indicies])

    for i, text in enumerate(x):
        ax.annotate(str(kmean_indicies[i]), (x[i], y[i]))
    plt.savefig(os.getcwd()+'/Results/kmeans.png')
    
    print("Plot saved the in Results Folder")
